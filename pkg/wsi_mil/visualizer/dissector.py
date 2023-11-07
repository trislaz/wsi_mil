from wsi_mil.deepmil.predict import load_model
from matplotlib.colors import Normalize
from scipy.special import softmax
from openslide import open_slide 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import skimage
from skimage import filters
import matplotlib.patches as patches
import torch
from argparse import Namespace
from ..tile_wsi.utils import get_image, get_x_y_from_0, get_size, get_whole_image
import matplotlib.pyplot as plt
from glob import glob
import yaml
from copy import copy
import numpy as np
import pickle
import pandas as pd
from .model_hooker import HookerMIL
from .utils import make_background_neutral, add_titlebox, set_axes_color
from pathlib import Path
import os 
from scipy.ndimage import zoom
from PIL import Image

class BaseVisualizer(ABC):
    def __init__(self, model, path_emb=None, path_raw=None):
        ## Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model, self.device)
        self.label_encoder = self.model.label_encoder

        # Par défaut on considère le dataset d'entrainement 
        # = données contenues dans args.
        self.table = self._load_table(self.model.table_data)
        self.path_emb = self.model.args.wsi if path_emb is None else path_emb
        self.path_emb_mat = os.path.join(self.path_emb)
        self.path_raw = self.model.args.raw_path if path_raw is None else path_raw
        self.num_class = self.model.args.num_class
        self.num_heads = self.model.args.num_heads
        self.target_name = self.model.args.target_name
        self.hooker = HookerMIL(self.model.network, self.model.args.num_heads)

    def _load_table(self, table):
        warning_msg = "         Carefull :          \n"
        warning_msg +="you are loading a table_data from path \n"
        warning_msg +="the test attribution of data might be different \n"
        warning_msg +="used during training.  \n"
        warning_msg +="Performances might be overestimated."
        if type(table) is str:
            print(warning_msg)
            table = pd.read_csv(table)
        return table

    def _get_info(self, wsi_ID, path_emb):
        wsi_info_path = os.path.join(path_emb, 'info')
        infomat = os.path.join(wsi_info_path, wsi_ID + '_infomat.npy')
        infomat = np.load(infomat)
        infomat = infomat.T 
        with open(os.path.join(wsi_info_path, wsi_ID+ '_infodict.pickle'), "rb") as f:
            infodict = pickle.load(f)
            infodict = self.add_name_to_dict(infodict, wsi_ID) 
        return {'paradict': infodict, "infomat" : infomat}
    
    def add_name_to_dict(self, dico, name):
        """
        because the name of the slide is not included in the dico.
        """
        for d in dico:
            dico[d]['name'] = name
        return dico

    def _get_image(self, path_raw, info):
        """_get_image.
        extract the indice-ieme tile of the wsi stored at path_raw.
        returns PIL image.

        :param path_raw: path to te wsi (all extensions accepted)
        :param indice: number of the tile in the flattened WSI
        """
        param = info
        path_wsi = glob(os.path.join(self.path_raw, param['name'] + '.*'))
        assert path_wsi, "no wsi with name {}".format(param['name'])
        assert len(path_wsi)<2, "several wsi with name {}".format(param['name'])
        slide = open_slide(path_wsi[0])
        image = slide.read_region(location=(param['x'], param['y']),
                                level=param['level'], 
                                size=(448, 448))#(param['xsize'], param['ysize']))
        return image

    def _preprocess(self, input_array, expand_bs=False):
        """preprocess the input to feed the model
    
        Args:
            input_path (str): str to the input path 
        """
        depth = self.model.args.feature_depth
        inp = input_array[:,:depth]
        #inp = self.model.ipca.transform(inp)
        inp = torch.Tensor(inp)
        inp = inp.unsqueeze(0) if expand_bs else inp
        inp = inp.to(self.device)
        return inp

    def set_data_path(self, path_raw=None, path_emb=None, table=None):
        self.path_raw = path_raw if path_raw is not None else self.path_raw
        self.path_emb = path_emb if path_raw is not None else self.path_emb
        self.table = table if table is not None else self.table

    @abstractmethod
    def forward(self, wsi_ID):
        pass

class TileSeeker(BaseVisualizer): 
    """
    Decision-based extraction of predictive tiles.
    Takes as input a model of MIL (AttentionMIL), extracts the predictive tiles
    contained in the train + test set used for training the model.
    """
    def __init__(self, model, n_best, min_prob=False, max_per_slides=None, path_emb=None, att_thres='otsu', store=True):
        """__init__.

        :param model: str, path to a DeepMIL checkpoint.
        :param n_best: int, number of tiles to keep per class.
        :param min_prob: if True, keeps tiles MINIMIZING a given logit.
        :param max_per_slides: diversity parameter. A WSI can not participate to 
        :param att_thres: 'str' or int: if str = 'otsu', use otsu threshold to select for tiles, 
        else, use setted number of tile threshold (usually 300)
        the selection with more than max_per_slides tiles.
        :param path_emb: str,  
        """
        super(TileSeeker, self).__init__(model, path_emb)
        self.classifier = self.model.network.mil.classifier
        self.classifier.eval()
        self.attention = self.model.network.mil.attention
        self.attention.eval()
        self.n_best = n_best
        self.min_prob = min_prob
        self.max_per_slides = max_per_slides
        self.att_thres = att_thres
        self.store = store
        self.path_raw = self.model.args.raw_path
        self.attention_scores = None
        self.decision_scores = None

        ## Model parameters
        assert self.model.args.num_heads == 1, 'you can\'t extract a best tile when using the multiheaded attention'
        self.model_name = self.model.args.model_name
        self.target_name = self.model.args.target_name

        ## list that have to be filled with various WSI
        self.store_info = None
        self.store_score = None
        self.store_tile = None
        self.store_images = None
        self.store_preclassif = None
        self.store_attention = None
        self._reset_storage()

    def forward(self, wsi_ID):
        """forward.
        Execute a forward pass through the MLP classifier. 
        Stores the n-best tiles for each class.
        :param wsi_ID: wsi_ID as appearing in the table_data.
        """
        if type(wsi_ID) is int:
            x = glob(os.path.join(self.path_emb, 'mat_pca', '*_embedded.npy'))[wsi_ID]
            wsi_ID = os.path.basename(x).split('_embedded')[0]
            x = np.load(x)
        elif isinstance(wsi_ID, np.ndarray):
            x = wsi_ID
        else:
            x = os.path.join(self.path_emb, 'mat_pca', wsi_ID+'_embedded.npy')
            x = np.load(x)

        info = self._get_info(wsi_ID, path_emb=self.path_emb)

        #process each images
        x = self._preprocess(x)
        out = self.classifier(x) # (bs, n_class)
        logits = self.hooker.scores

        self.attention(x.unsqueeze(0))
        tw = self.hooker.tiles_weights.squeeze()
        ## Otsu thresholding
        if self.att_thres == 'otsu':
            otsu = filters.threshold_otsu(tw)
            selection = np.where(tw >= otsu)[0]
        elif isinstance(self.att_thres, int):
            _, ind = torch.sort(torch.Tensor(tw))
            size_select = min(thres_otsu, len(ind))
            selection = set(ind[-size_select:].cpu().numpy())

        self.attention_scores = tw
        self.decision_scores = logits
        
        # Find attention scores to filter out of distribution tiles
        if self.store:
            self.store_best(x.cpu().numpy(), logits, info, selection, min_prob=self.min_prob, max_per_slides=self.max_per_slides)
        return self

    def forward_all(self):
        table = self.table
        wsi_ID = table['ID'].values
        for n, o in enumerate(wsi_ID):
            self.forward(o)
        return self

    def store_best(self, inp, out, info, selection_att, min_prob=False, max_per_slides=None):
        """store_best.
        decides if we have to store the tile, according the final activation value.

        :param out: out of a forward parss
        :param info: info dictionnary of the WSI
        :param min_prob: bool:
        maximiser proba -> prendre les derniers éléments de indices_best (plus grand au plus petit)
        minimiser proba -> prendre les premiers éléments de indice_best
        """
        sgn = -1 if min_prob else 1
        # for each tile
        tmp_infostore, tmp_scorestore , tmp_tilestore, tmp_attentionstore, tmp_preclassifstore = dict(), dict(), dict(), dict(), dict()

        for o,i in enumerate(self.label_encoder.classes_):
            tmp_scorestore[o] = []
            tmp_infostore[o] = []
            tmp_tilestore[o] = []
            tmp_attentionstore[o] = []
            tmp_preclassifstore[o] = []

        for s in range(out.shape[0]): 
            if s not in selection_att:
                continue
            # for each class
            for o in range(len(self.label_encoder.classes_)):
                # If the score for class o at tile s is bigger than the smallest 
                # stored value: put in storage
                if (len(self.store_score[o]) < self.n_best) or (sgn * out[s,o] >= sgn * self.store_score[o][0]):
                    tmp_infostore[o].append(info['paradict'][s])
                    tmp_tilestore[o].append(inp[s,:])
                    tmp_attentionstore[o].append(self.hooker.tiles_weights[s, :])
                    tmp_preclassifstore[o].append(self.hooker.reprewsi[s, :])
                    tmp_scorestore[o].append(out[s,o])

        ## Selects the n-best tiles per WSI.
        for o,i in enumerate(self.label_encoder.classes_): 
            selection = np.argsort(tmp_scorestore[o])[::-sgn][:max_per_slides] 
            self.store_info[o] += list(np.array(tmp_infostore[o])[selection])
            self.store_score[o] += list(np.array(tmp_scorestore[o])[selection])
            self.store_tile[o] += list(np.array(tmp_tilestore[o])[selection])
            self.store_attention[o] += list(np.array(tmp_attentionstore[o])[selection])
            self.store_preclassif[o] += list(np.array(tmp_preclassifstore[o])[selection])

        for o in range(len(self.label_encoder.classes_)):
            indices_best = np.argsort(self.store_score[o])[::sgn][-self.n_best:]
            self.store_score[o] = list(np.array(self.store_score[o])[indices_best])
            self.store_tile[o] = list(np.array(self.store_tile[o])[indices_best])
            self.store_info[o] = list(np.array(self.store_info[o])[indices_best])
            self.store_attention[o] = list(np.array(self.store_attention[o])[indices_best])
            self.store_preclassif[o] = list(np.array(self.store_preclassif[o])[indices_best])

    def _reset_storage(self):
        """_reset_storage.
        Reset the storage dict.
        store_score and store info are dict with keys the classes (ordinals)
        containing empty lists. When filled, they are supposed to n_best scores 
        and infodicts values.
        only store images as the name of the targets as keys. 
        Advice : fill store image at the end only.
        """
        self.store_score = dict()
        self.store_info = dict()
        self.store_tile = dict()
        self.store_image = dict()
        self.store_preclassif = dict()
        self.store_attention = dict()
        for o,i in enumerate(self.label_encoder.classes_):
            self.store_info[o] = []
            self.store_score[o] = []
            self.store_tile[o] = []
            self.store_image[i] = []
            self.store_preclassif[o] = []
            self.store_attention[o] = []

    def extract_images(self):
        for o,i in enumerate(self.label_encoder.classes_):
            assert self.store_score[o], "no tile found"
            self.store_image[i] = [self._get_image(self.path_raw, x).convert('RGB') for x in self.store_info[o]]
        return self

class ConsensusTileSeeker(TileSeeker):
    def __init__(self, model, n_best, min_prob=False, max_per_slides=None, path_emb=None, att_thres='otsu'):
        super(ConsensusTileSeeker, self).__init__(model[0], n_best, min_prob, max_per_slides, path_emb, att_thres, False)
        tile_seekers = []
        for m in model:
            ts = TileSeeker(m, n_best, min_prob, max_per_slides, path_emb, att_thres, False)
            tile_seekers.append(ts)
        self.seekers = tile_seekers
        self.model_name = tile_seekers[0].model.args.model_name
        self.target_name = tile_seekers[0].model.args.target_name

        ## list that have to be filled with various WSI
        self.store_info = None
        self.store_score = None
        self.store_tile = None
        self.store_images = None
        self.store_preclassif = None
        self.store_attention = None
        self._reset_storage()

    def forward(self, wsi_ID):
        """forward.
        Execute a forward pass through the MLP classifier. 
        Stores the n-best tiles for each class.
        :param wsi_ID: wsi_ID as appearing in the table_data.
        """
        if type(wsi_ID) is int:
            x = glob(os.path.join(self.path_emb,'mat_pca', '*_embedded.npy'))[wsi_ID]
            wsi_ID = os.path.basename(x).split('_embedded')[0]
            x = np.load(x)
        elif isinstance(wsi_ID, np.ndarray):
            x = wsi_ID
        else:
            x = os.path.join(self.path_emb,  'mat_pca', wsi_ID+'_embedded.npy')
            x = np.load(x)

        info = self._get_info(wsi_ID, path_emb=self.path_emb)

        #process each images
        x = self._preprocess(x)
        outs = []
        logits = []
        attention = []
        for s in self.seekers:
            outs.append(s.classifier(x).detach().cpu().numpy())
            logits.append(s.hooker.scores)
            s.attention(x.unsqueeze(0))
            attention.append(s.hooker.tiles_weights)
        out = np.mean(outs, 0)
        logits = np.mean(logits, 0)
        tw = np.mean(attention, 0)
        #filling the hooker with mean values
        self.hooker.tiles_weights = tw
        self.hooker.reprewsi = s.hooker.reprewsi

        ## Otsu thresholding
        if self.att_thres == 'otsu':
            otsu = filters.threshold_otsu(tw)
            selection = np.where(tw >= otsu)[0]
        elif isinstance(self.att_thres, int):
            _, ind = torch.sort(torch.Tensor(tw))
            size_select = min(thres_otsu, len(ind))#int(len(ind)/100)
            selection = set(ind[-size_select:].cpu().numpy())
        
        # Find attention scores to filter out of distribution tiles
        self.store_best(x.cpu().numpy(), logits, info, selection, min_prob=self.min_prob, max_per_slides=self.max_per_slides)
        return self

class HeatmapMaker(BaseVisualizer):
    def __init__(self, model, path_emb=None, path_raw=None):
        model = list(Path(model).glob('*.pt.tar'))
        super(HeatmapMaker, self).__init__(model[0], path_emb, path_raw)
        tile_seekers = []
        for m in model:
            ts = TileSeeker(m, 0, 0, 0, path_emb, 0, False)
            tile_seekers.append(ts)
        self.seekers = tile_seekers
        self.model_name = tile_seekers[0].model.args.model_name
        self.target_name = tile_seekers[0].model.args.target_name
        self.path_emb = tile_seekers[0].path_emb

    def forward(self, wsi_ID):
        """forward.
        Execute a forward pass through the MLP classifier. 
        Stores the n-best tiles for each class.
        :param wsi_ID: wsi_ID as appearing in the table_data.
        """
        if type(wsi_ID) is int:
            x = glob(os.path.join(self.path_emb,'mat_pca', '*_embedded.npy'))[wsi_ID]
            wsi_ID = os.path.basename(x).split('_embedded')[0]
            x = np.load(x)
        elif isinstance(wsi_ID, np.ndarray):
            x = wsi_ID
        else:
            x = os.path.join(self.path_emb,  'mat_pca', wsi_ID+'_embedded.npy')
            x = np.load(x)
        
        info = self._get_info(wsi_ID, path_emb=self.path_emb)
        
        #process each images
        x = self._preprocess(x)
        outs = []
        logits = []
        attention = []
        for s in self.seekers:
            outs.append(s.classifier(x).detach().cpu().numpy())
            logits.append(s.hooker.scores)
            s.attention(x.unsqueeze(0))
            attention.append(s.hooker.tiles_weights)
        out = np.mean(outs, 0)
        logits = np.mean(logits, 0)
        tw = np.squeeze(np.mean(attention, 0))
        #filling the hooker with mean values
        return tw, logits, info['infomat']

    def make_heatmap(self, wsi_ID, downsample_factor=16):
        """
        Generates and saves a heatmap overlay image for the given whole-slide image ID.
        
        Args:
        wsi_ID (str): The identifier for the whole-slide image.
        """
        heatmaps_per_class = {}
        tw, logits, infomat = self.forward(wsi_ID)
        for class_idx in range(logits.shape[1]):
            # Softmax the attention scores
            maxtw = softmax(tw)
            heatmap_scores = maxtw * logits[:, class_idx]
            heatmaps_per_class[self.label_encoder.classes_[class_idx]], background = self.create_heatmap(infomat, heatmap_scores)

        max_over_classes = np.max(np.stack([heatmap for heatmap in heatmaps_per_class.values()]))
        min_over_classes = np.min(np.stack([heatmap for heatmap in heatmaps_per_class.values()]))

        for target in self.label_encoder.classes_:
            overlay = self.overlay_heatmap_on_thumbnail(wsi_ID, heatmaps_per_class[target], downsample_factor, background, max_over_classes, min_over_classes)
            overlay.save(f"{wsi_ID}_class_{target}_heatmap.png")

        # Attention HM:
        heatmap_normalized, background = self.create_heatmap(infomat, tw)
        overlay = self.overlay_heatmap_on_thumbnail(wsi_ID, heatmap_normalized, downsample_factor, background, np.max(heatmap_normalized), np.min(heatmap_normalized))
        overlay.save(f"{wsi_ID}_attention_heatmap.png")

        
    def create_heatmap(self, infomat, heatmap_scores):
        """
        Creates a normalized heatmap from scores and an infomat matrix.
        
        Args:
        infomat (np.ndarray): A matrix mapping tile indices to their position.
        heatmap_scores (np.ndarray): An array containing the scores for each tile.
        
        Returns:
        np.ndarray: A normalized heatmap as a 2D numpy array.
        """
        heatmap = np.zeros_like(infomat, dtype=np.float32)
        unique_tiles = np.unique(infomat)
        for tile in unique_tiles:
            if tile >= 0:
                tile = int(tile)
                heatmap[infomat == tile] = heatmap_scores[tile]
        background = infomat == -1
        return heatmap, background
    
    def overlay_heatmap_on_thumbnail(self, wsi_ID, heatmap, downsample_factor, background, max_over_classes, min_over_classes):
        """
        Overlays the normalized heatmap onto a higher magnification thumbnail of the whole-slide image and saves it.
        
        Args:
        wsi_ID (str): The identifier for the whole-slide image.
        heatmap_normalized (np.ndarray): A normalized heatmap as a 2D numpy array.
        downsample_factor (int): The factor by which the heatmap should be downsampled for overlay.
        """
        wsi_path = glob(os.path.join(self.path_raw, wsi_ID + '.*'))[0]
        wsi_image = open_slide(wsi_path)
        level = wsi_image.get_best_level_for_downsample(downsample_factor)
        thumbnail = wsi_image.read_region((0, 0), level, wsi_image.level_dimensions[level])
        # Thumbnail in grey values
        thumbnail = thumbnail.convert("L").convert('RGB')

        scale_factor_y = thumbnail.size[1] / heatmap.shape[0]
        scale_factor_x = thumbnail.size[0] / heatmap.shape[1]

        # Normalize the heatmap to the desired range.
        norm = Normalize(vmin=min_over_classes, vmax=max_over_classes)
        heatmap_normalized = norm(heatmap)
        # heatmap_normalized = heatmap

        heatmap_resized = zoom(heatmap_normalized, (scale_factor_y, scale_factor_x), order=0) # order = 0 if you want to keep the original tile geometry
        #heatmap_resized = skimage.transform.resize(heatmap_normalized.T, thumbnail.size).T
        background_resized = zoom(background, (scale_factor_y, scale_factor_x), order=0)

        # Apply the colormap only on non-background pixels
        colormap = plt.get_cmap('coolwarm')

        heatmap_filtered = np.ma.masked_array(heatmap_resized, mask=background_resized)
        colored_heatmap = colormap(heatmap_filtered, bytes=True)
        colored_heatmap[..., :3][background_resized] = 0
        colored_heatmap_pil = Image.fromarray(colored_heatmap[..., :3])

        ## Make the background neutral
        neutral_color = [0, 0, 0]  # Gray color for neutral background
        neutral_background = Image.new("RGB", colored_heatmap_pil.size, color=tuple(neutral_color))
        mask = Image.fromarray((~background_resized).astype(np.uint8) * 255)  # Invert mask for correct transparency
        combined_heatmap = Image.composite(colored_heatmap_pil, neutral_background, mask)
        thumbnail_w_background = Image.composite(thumbnail, neutral_background, mask)
        combined_heatmap = Image.blend(thumbnail_w_background, combined_heatmap, alpha=0.65)
        return combined_heatmap









