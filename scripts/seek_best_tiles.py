from wsi_mil.visualizer.dissector import TileSeeker
from argparse import ArgumentParser
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--n_best', type=int, default=20)
parser.add_argument('--min_prob', action='store_true')
parser.add_argument('--max_per_slides', type=int, default=None)
parser.add_argument('--only_test', action='store_true')
args = parser.parse_args()

ts = TileSeeker(model=args.model, n_best=args.n_best, min_prob=args.min_prob, max_per_slides=args.max_per_slides)
ts.forward_all(only_test=args.only_test)
ts.extract_images()
out = os.path.join(os.path.dirname(args.model), 'summaries_{}'.format(os.path.basename(args.model).split('.pt.tar')[0]), 'hightiles')
out_encoded = out + '_encoded'
os.makedirs(out_encoded, exist_ok=True)
for k in ts.store_image:
    os.makedirs(os.path.join(out, str(k)), exist_ok=True)
    [x.save(os.path.join(out, str(k), 'tile_{}.jpg'.format(o))) for o,x in enumerate(ts.store_image[k])]

for o, n in enumerate(ts.store_tile):
    var = ts.label_encoder.classes_[o]
    tiles = ts.store_tile[n]
    tiles = np.vstack(tiles)
    np.save(os.path.join(out_encoded, str(var)+'.npy'), tiles)
    attention = np.vstack(ts.store_attention[o])
    preclassif = np.vstack(ts.store_preclassif[o])
    scores = np.vstack(ts.store_score[o])
    np.save(os.path.join(out_encoded, str(var)+'_attention.npy'), attention)
    np.save(os.path.join(out_encoded, str(var)+'_preclassif.npy'), preclassif)
    np.save(os.path.join(out_encoded, str(var)+'_scores.npy'), scores)


