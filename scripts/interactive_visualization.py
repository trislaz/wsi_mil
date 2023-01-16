import numpy as np
from umap import UMAP
from glob import glob
from argparse import ArgumentParser
import hdbscan
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import preprocessing
import seaborn as sns
import numpy as np; np.random.seed(42)
from argparse import ArgumentParser
import os
import pandas as pd
from PIL import Image

def set_axes_color(ax, color='orange'):
    dirs = ['bottom', 'top', 'left', 'right']
    args = {x:False for x in dirs}
    args.update({'label'+x:False for x in dirs})
    args.update({'axis':'both', 'which': 'both'})
    ax.tick_params(**args)
    for sp in ax.spines:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(2)
    return ax

def main(projection, path_im, path_size=None, selection=None, palette='magma_r'):
    if selection is None:
        selection = list(range(projection.shape[0]))
    le = preprocessing.LabelEncoder()
    if path_size is not None:
        if type(path_size) is str:
            att = np.squeeze(np.load(path_size))
            size = [int(att[selection[o]]) for o,x in enumerate(att)]
            alpha = [x/np.max(att) for x in att]
        else:
            size = [float(path_size[o]) for o in selection]
    else:
        size = 40
    cmap = plt.cm.RdYlGn
    gridsize = (2, 3)
    image_path = np.asarray([glob('{}/{}_*.jpg'.format(path_im, o))[0] for o in range(projection.shape[0])])
    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot2grid(gridsize, (0,0), rowspan=2, colspan=2, fig=fig)
    ax.set_navigate(True)
    global axes_im
    axes_im = [plt.subplot2grid(gridsize, (o, 2), rowspan=1, colspan=1, fig=fig) for o in range(2)]
    [(x.set_xticks([]), x.set_yticks([]), x.set_navigate(False), x.imshow(plt.imread(image_path[0]))) for x in axes_im]
    line = sns.scatterplot(x=projection[:,0], y=projection[:,1], size=50, alpha=0.7,hue=size, ax=ax, palette=palette)
    line = line.collections[0]
    global counter
    counter = 0
    def hover(event):
        global counter
        if line.contains(event)[0]:
            counter += 1
            ind, = line.contains(event)[1]["ind"]
            global axes_im
            axes_im = [set_axes_color(ax, 'r') if o == counter%2 else set_axes_color(ax, 'black') for o,ax in enumerate(axes_im)]
            axes_im[counter%2].imshow(plt.imread(image_path[ind]))
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', hover)
    fig = plt.gcf()
    fig.set_size_inches(5, 4.5)
    plt.show()


parser = ArgumentParser()
parser.add_argument('--path', type=str, help='path of the visualization outputs')
parser.add_argument('--target_val', type=str, help='See tiles corresponding to a particular value of the target variable. Coincide with the names of the folders in path/hightiles/' )
parser.add_argument('--decision_emb', action='store_true', help='when tagged, use the decision embeddings (related to the target variable).')
parser.add_argument('--cluster', action='store_true', help='If tagged, tiles will be clustered. If not, color of the points corresponds to the decision score(tiles logits)')
args = parser.parse_args()

um = UMAP(n_components=2, n_neighbors=50, min_dist=0)
if args.decision_emb:
    emb_path = os.path.join(args.path, 'hightiles_encoded', f'{args.target_val}_preclassif.npy')
else:
    emb_path = os.path.join(args.path, 'hightiles_encoded', f'{args.target_val}.npy')
M = um.fit_transform(np.load(emb_path))
labels = hdbscan.HDBSCAN(min_samples=3,min_cluster_size=30).fit_predict(M)
scores = np.load(os.path.join(args.path, 'hightiles_encoded', f'{args.target_val}_scores.npy'))
colors, palette = (labels, 'bright') if args.cluster else (scores, 'magma_r')
main(M, os.path.join(args.path, f'hightiles/{args.target_val}'), path_size = labels, palette='bright')
