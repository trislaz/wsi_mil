from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
from joblib import load, dump
import os

def check_dim(batch):
    """ Checks if batch is big enough for the incremental PCA to be 
    efficient.
    
    Parameters
    ----------
    batch : list
        list of matrix, each matrix corresponding to a WSI divided in $row tiles
    
    Returns
    -------
    bool
        Is the batch big enough ?
    """
    if batch:
        n_tiles = np.sum([x.shape[0] for x in batch])
        n_features = batch[-1].shape[1]
        ans = n_tiles >= n_features
    else:
        ans = False
    return ans

def get_files(path):
    files = glob(os.path.join(path, 'mat', '*.npy'))
    return files

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default=".", help="path to the files of tiles")
    args = parser.parse_args(raw_args)
    files = get_files(args.path) 
    ipca = IncrementalPCA()
    batch = []
    for path in tqdm(files):
        mat = np.load(path)
        if len(mat.shape) == 1:
            mat = np.expand_dims(mat, 0)
        if mat.sum() == 0:
            continue
        if check_dim(batch):
            batch = np.vstack(batch)
            ipca.partial_fit(X=batch)
            batch = []
        else:
            batch.append(mat)

    msg = " ----------------  RESULTS -------------------- \n"
    s = 0
    for i,o in enumerate(ipca.explained_variance_ratio_, 1):
        s += o
        msg += "Dimensions until {} explains {}% of the variance \n".format(i, s*100)
    msg += "----------------------------------------------"

    ## Saving
    with open('./pca/results.txt', 'w') as f:
        f.write(msg)

    dump(ipca, 'pca/pca_tiles.joblib')
    return ipca
