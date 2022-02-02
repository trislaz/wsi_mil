from glob import glob
from argparse import ArgumentParser
from torchvision.models import resnet50, resnet18, resnext50_32x4d
from torchvision import transforms
from torch.nn import Identity
import torch
import pandas as pd
import pickle
import os
import numpy as np
from PIL import Image
from xml.dom import minidom
from skimage.draw import polygon
from skimage.morphology import dilation
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage._shared.utils import warn
import openslide

from .utils import make_auto_mask, patch_sampling, get_size, visualise_cut, get_image

#TODO Ajouter name_slide dans les infos

class ImageTiler:
    """
    Class implementing several possible tiling.
    initialized with the namespace args:
        args    .path_wsi: path to the WSI to tile.
                .path_mask: path to the annotation in xml format. Not necessary 
                    if auto_mask= 1.
                .level: pyramidal level of tile extraction.
                .size: dimension in pixel of the extracted tiles.
                .auto_mask: 0 or 1. If 1, automatically extracts relevant
                    portions of the WSI
                .tiler: type of tiling. available: simple | imagenet | moco
                .path_outputs: str, root of the paths where are stored the outputs.
                .model_path: str, when using moco tiler.
                .mask_tolerance: minimum percentage of mask on a tile for selection.
    """
    def __init__(self, args, make_info=True):
        self.level = args.level # Level to which sample patch.
        self.nf = args.nf # Useful for managing the outputs
        self.device = args.device
        self.size = (args.size, args.size)
        self.path_wsi = args.path_wsi 
        self.max_nb_tiles = args.max_nb_tiles
        self.path_outputs = os.path.join(args.path_outputs, args.tiler, f'level_{args.level}')
        self.auto_mask = args.auto_mask
        self.path_mask = args.path_mask
        self.model_path = args.model_path 
        self.infomat = None
        self.tiler = args.tiler
        self.name_wsi, self.ext_wsi = os.path.splitext(os.path.basename(self.path_wsi))
        # If nf is used, it manages the output paths.
        self.outpath = self._set_out_path()
        self.slide = openslide.open_slide(self.path_wsi)
        self.make_info = make_info
        if args.mask_level < 0:
            self.mask_level = self.slide.level_count + args.mask_level
        else:
            self.mask_level = args.mask_level    
        self.rgb_img = self.slide.get_thumbnail(self.slide.level_dimensions[self.mask_level]) 
        self.rgb_img = np.array(self.rgb_img)[:,:,:3]
        self.mask_tolerance = args.mask_tolerance

    def _set_out_path(self):
        """_set_out_path. Sets the path to store the outputs of the tiling.
        Creates them if they do not exist yet.
        """
        outpath = dict()
        nf = self.nf
        tiler = self.tiler
        outpath['info'] = '.' if nf else os.path.join(self.path_outputs, 'info') 
        outpath['visu'] = '.' if nf else os.path.join(self.path_outputs, 'visu')
        if tiler == 'simple':
            outpath['tiles'] = '.' if nf else os.path.join(self.path_outputs, self.name_wsi)
        else:
            outpath['tiles'] = '.' if nf else os.path.join(self.path_outputs, 'mat')
        #Creates the dirs.
        [os.makedirs(v, exist_ok=True) for k,v in outpath.items()]
        return outpath
 
    def _get_mask_function(self):
        """
        the patch sampling functions need as argument a function that takes a WSI a returns its 
        binary mask, used to tile it. here it is.
        """
        if self.auto_mask:
            mask_function = lambda x: make_auto_mask(x, mask_level=-1)
        else:
            path_mask = os.path.join(self.path_mask, self.name_wsi + ".xml")
            assert os.path.exists(path_mask), "No annotation at the given path_mask"
            mask_function = lambda x: get_polygon(image=self.rgb_img, path_xml=path_mask, label='t')
        return mask_function

    def tile_image(self):
        """tile_image.
        Main function of the class. Tiles the WSI and writes the outputs.
        WSI of origin is specified when initializing TileImage.
        """
        self.mask_function = self._get_mask_function()
        tiler = getattr(self, self.tiler + '_tiler')
        param_tiles = patch_sampling(slide=self.slide, mask_level=self.mask_level, mask_function=self.mask_function, 
            analyse_level=self.level, patch_size=self.size, mask_tolerance = self.mask_tolerance)
        if self.make_info:
            self._make_infodocs(param_tiles)
            self._make_visualisations(param_tiles)
        tiler(param_tiles)

    def _make_visualisations(self, param_tiles):
        """_make_visualisations.
        Creates and save an image showing the locations of the extracted tiles.

        :param param_tiles: list, output of usi.patch_sampling.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use('Agg')
        PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                     'title': "n_tiles={}".format(len(param_tiles))}
        visualise_cut(self.slide, param_tiles, res_to_view=self.mask_level, plot_args=PLOT_ARGS)
        plt.savefig("{}_visu.png".format(os.path.join(self.outpath['visu'], self.name_wsi)))

    def _get_infomat(self):
        """Returns a zero-matrix, such that each entry correspond to a tile in the WSI.
        ID of each tiles will be stored here. 

        Returns
        -------
        tuple
            (mat -ndarray- , size_patch_0 -int, size of a patch in level 0- )
        """
        size_patch_0 = get_size(self.slide, size_from=self.size, level_from=self.level, level_to=0)[0]
        dim_info_mat = (self.slide.level_dimensions[0][0] // size_patch_0, self.slide.level_dimensions[0][1] // size_patch_0)
        info_mat = np.zeros(dim_info_mat)
        return info_mat, size_patch_0 

    def _make_infodocs(self, param_tiles):
        """_make_infodocs.
        Creates the files containing the information relative 
        to the extracted tiles.
        infos are saved in the outpath 'info'.
        *   infodict : same as param_tiles, but dictionnary version (needs pickle 
            load them). Stores the ID of the tile, the x and y position in the 
            original WSI, the size in pixels at the desired level of extraction and 
            finally the level of extraction.
        *   infodf : same as infodict, stored as a pandas DataFrame.
        *   infomat : relates the tiles to their position on the WSI.
            matrix of size (n_tiles_H, n_tiles_W), each cell correspond to a tile
            and is fill with -1 if the tile is background else with the tile ID.

        :param param_tiles: list: output of the patch_sampling.
        """
        infodict = {}
        infos=[]
        infomat , patch_size_0 = self._get_infomat() 
        if self.tiler == 'classifier':
            self.infomat_classif = np.zeros(infomat.shape)
        for o, para in enumerate(param_tiles):
            infos.append({'ID': o, 'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]})
            infodict[o] = {'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]} 
            infomat[para[0]//(patch_size_0+1), para[1]//(patch_size_0+1)] = o + 1 
            #+1 car 3 lignes plus loin je sauve infomat-1 (background à -1) 
        df = pd.DataFrame(infos)
        self.infomat = infomat - 1 

        # Saving
        df.to_csv(os.path.join(self.outpath['info'], self.name_wsi + '_infos.csv'), index=False)
        np.save(os.path.join(self.outpath['info'], self.name_wsi + '_infomat.npy'), infomat-1)
        with open(os.path.join(self.outpath['info'],  self.name_wsi + '_infodict.pickle'), 'wb') as f:
            pickle.dump(infodict, f)
   
    def simple_tiler(self, param_tiles):
        """simple_tiler.
        Simply writes tiles as .png

        :param param_tiles: list: output of the patch_sampling.
        """
        if self.max_nb_tiles is not None:
            n = min(self.max_nb_tiles, len(param_tiles))
            param_tiles = np.array(param_tiles)[np.random.choice(range(len(param_tiles)), n, replace=False)]
        for o, para in enumerate(param_tiles):
            patch = get_image(slide=self.path_wsi, para=para, numpy=False)
            path_tile = os.path.join(self.outpath['tiles'], f"tile_{o}.png")
            patch.save(path_tile)
            del patch

    def _forward_pass_WSI(self, model, param_tiles, preprocess):
        """_forward_pass_WSI. Feeds a pre-trained model, already loaded, 
        with the extracted tiles.

        :param model: Pytorch loaded module.
        :param param_tiles: list, output of the patch_sampling.
        :param preprocess: torchvision.transforms.Compose.
        """
        tiles = []
        for o, para in enumerate(param_tiles):
            image = get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
            tiles.append(t.cpu().numpy())
        mat = np.vstack(tiles)
        return mat

    def imagenet_tiler(self, param_tiles):
        """imagenet_tiler.
        Encodes each tiles thanks to a resnet18 pretrained on Imagenet.
        Embeddings are 512-dimensionnal.

        :param param_tiles: list, output of the patch_sampling.
        """
        model = resnet18(pretrained=True)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = self._get_transforms(imagenet=True)
        mat = self._forward_pass_WSI(model, param_tiles, preprocess)
        np.save(os.path.join(self.outpath['tiles'], f'{self.name_wsi}_embedded.npy'), mat)

    def _get_transforms(self, imagenet=True):
        """_get_transforms.
        For tiling encoding, normalize the input with moments of the
        imagenet distribution or of the training set of the MoCo model.

        :param imagenet: bool: use imagenet pretraining.
        """
        if not imagenet:
            trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.747, 0.515,  0.70], [0.145, 0.209, 0.154])])
        else: 
            trans = transforms.Compose([transforms.ToTensor(), 
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return trans

    def ciga_tiler(self, param_tiles):
        def load_model_weights(model, weights):
            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)
            return model
        model = resnet18()
        state = torch.load(self.model_path, map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        tiles = []
        #param_tiles = np.array(param_tiles)[np.random.choice(range(len(param_tiles)), min(4000,len(param_tiles)), replace=False)]
        for o, para in enumerate(param_tiles):
            image = usi.get_image(slide=self.slide, para=para, numpy=False)
            image = image.convert("RGB")
            if self.from_0:
                image = image.resize(self.size)
            image = preprocess(image).unsqueeze(0)
            image = image.to(self.device)
            with torch.no_grad():
                t = model(image).squeeze()
            tiles.append(t.cpu().numpy())
        mat = np.vstack(tiles)
        np.save(os.path.join(self.path_outputs, '{}_embedded.npy'.format(self.name_wsi)), mat)



    def moco_tiler(self, param_tiles):
        """moco_tiler.
        Encodes each tiles thanks to a resnet18 pretrained with MoCo.

        Code for loading the model taken from the MoCo official package:
        https://github.com/facebookresearch/moco

        :param param_tiles: list, output of the patch_sampling.
        """
        model = resnet50()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.fc = Identity()
        model = model.to(self.device)
        model.eval()
        preprocess = self._get_transforms(imagenet=False)
        mat = self._forward_pass_WSI(model, param_tiles, preprocess)
        np.save(os.path.join(self.outpath['tiles'], 'mat', f'{self.name_wsi}_embedded.npy'), mat)

    def simclr_tiler(self, param_tiles):
        raise NotImplementedError
