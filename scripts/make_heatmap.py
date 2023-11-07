from wsi_mil.visualizer.dissector import HeatmapMaker
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--models_path', type=str, default='/path/with/models')
parser.add_argument('--path_emb', type=str, help="path the the embeddings = root directory containing mat_pca/ and info/", default=None)
parser.add_argument('--path_raw', type=str, help="path to the WSI images in pyramidal format", default=None)
parser.add_argument('--wsi_ID', type=str, help="name of the WSI to be processed", default=None)
parser.add_argument('--ds', type=int, help="downsampling factor of the heatmaps", default=16)
args = parser.parse_args()

hm = HeatmapMaker(model=args.models_path, path_emb=args.path_emb, path_raw=args.path_raw)
hm.make_heatmap(args.wsi_ID, downsample_factor=args.ds)
