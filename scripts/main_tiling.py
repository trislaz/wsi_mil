from wsi_mil.tile_wsi.tiler import ImageTiler
from wsi_mil.tile_wsi.pca_partial import main as incremental_pca
from wsi_mil.tile_wsi.arguments import get_arguments
import os
import numpy as np
from glob import glob

args = get_arguments()
outpath = os.path.join(args.path_outputs, args.tiler, f'level_{args.level}')
if os.path.isfile(args.path_wsi):
    it = ImageTiler(args=args)
    it.tile_image()
else:
    dirs = []
    extensions = set(['.ndpi', '.svs', '.tif'])
    all_files = glob(os.path.join(args.path_wsi, '*.*'))
    files = [f for f in all_files if os.path.splitext(f)[1] in extensions]
    assert len(files) > 0, f"No files with extension {extensions} in {args.path_wsi}"
    for f in files:
        args.path_wsi = f
        it = ImageTiler(args=args)
        it.tile_image()

#PCA
if args.tiler != 'simple':
    os.chdir(outpath)
    os.makedirs('pca', exist_ok=True)
    raw_args = ['--path', '.']
    ipca = incremental_pca(raw_args)
    os.makedirs(os.path.join('./mat_pca'), exist_ok=True)
    files = glob('./mat/*.npy')
    for f in files:
        m = np.load(f)
        mp = ipca.transform(m)
        np.save(os.path.join('mat_pca', os.path.basename(f)), mp)
