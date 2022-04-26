from wsi_mil.tile_wsi.tiler import ImageTiler
from wsi_mil.tile_wsi.arguments import get_arguments
import pandas as pd
import os
from glob import glob

args = get_arguments()
df = pd.read_csv('~/work/data/diags_tcga_manifest.txt', sep='\t')
diags = set([os.path.join(args.path_wsi, x) for x in df['filename'].values])
job_id=int(os.environ["SLURM_ARRAY_TASK_ID"])
slides = set(glob(os.path.join(args.path_wsi, '*.svs')))
possible = list(slides.intersection(diags) - alreadyDone)
for ind in range(10):
    args.path_wsi = os.path.join(args.path_wsi, possible[job_id+ind])
    print(args.path_wsi)
    it = ImageTiler(args=args)
    it.tile_image()
