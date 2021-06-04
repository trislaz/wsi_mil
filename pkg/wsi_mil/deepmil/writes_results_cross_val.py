"""Writes results from the output directory of the 
train process in simple_cross_val.nf. On the contrary to hyperparameter search, 
no config params !
"""
from collections import MutableMapping 
from argparse import ArgumentParser
import numpy as np
from glob import glob
import pandas as pd
import os
import torch
import shutil

def extract_config(config_path):
    """extracts the number of the config from its path (/../config_n.yaml)
    return int
    """
    config, _ = os.path.splitext(os.path.basename(config_path))
    config = int(config.split('_')[1])
    return config

def extract_references(args):
    """extracts a dictionnary with the parameters of a run
    return dict
    """
    t = args.test_fold
    r = args.repeat
    ref = {'test': t, 'repeat': r}
    return ref

def convert_flatten(d, parent_key='', sep='_'):
    """
    Flattens a nested dict.
    Code taken from https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mean_dataframe(df):
    """Computes mean metrics for a set of repeat.
    for a given config c and a given test set t, computes
    1/r sum(metrics) over the repetitions.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of results. Columns contain config, test and repeat.
    
    returns:
    ----------
    df_mean_r = pd.DataFrame
        mean dataframe, w/o column repeat.
    df_mean_rt = pdf.DataFrame
        mean dataframe over repeats and then over test sets.
        mean of metrics for each config.
    """
    tests = set(df['test'])
    rows_r = []
    rows_rt = []
    rows_t = []
    for t in tests:
        dft = df[df['test'] == t]
        dft_m = dft.mean(axis=0)
        dft_m = dft_m.drop('repeat').to_frame().transpose()
        rows_r.append(dft_m)
        rows_t.append(dft_m)
    df_mean_t = pd.concat(rows_t, ignore_index=True)
    df_mean_t = df_mean_t.drop('test', axis=1)
    df_mean_r = pd.concat(rows_r, ignore_index=True)
    return df_mean_r, df_mean_t

def select_best_repeat(df, sgn_metric, ref_metric, path, n_best=1):
    """Selects, for a given config (best_config), the models
    that led to the best validation results = single run.
    Attention, best result is here the highest result. 
    (wont work when using the loss f.i)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with all results in it
    sgn_metric : int
        1 or -1. Help select the 'best' sample. Defautl is considering 
        the best as the lowest according to the ref_metric. 
    ref_metric : str
        metric on which to select 
    path : str
        root path to the experiment.

    Returns
    -------
    list
        list containing tuples of parameters (config, test, repeat)
        that unequivocately leads to a model.
    """
    tests = set(df['test'])
    selection = []
    for t in tests:
        df_t = df[df['test'] == int(t)]
        metric_t = sgn_metric * np.array(df_t[ref_metric])
        best_indices = np.array(np.argsort(metric_t)[:n_best])
        best_rep = df.loc[best_indices, 'repeat']
        best_rep = [int(x) for x in best_rep]
        selection.append(zip([int(t)] * n_best, best_rep))
    return selection

def copy_best_to_root(path, param):
    """Copy the best models for all the test_sets,
    and the config file in the root path of the experiment.
    if cross_val : just testing a single config. therefore no copy to do
    """
    for p in param:
        t, r = p
        model_path = os.path.join(path, "test_{}/rep_{}/model_best.pt.tar".format(t, r))
        model_path = os.path.abspath(model_path)
        shutil.copy(model_path, 'model_best_test_{}_repeat_{}.pt.tar'.format(t, r))

def main(raw_args=None):
    parser = ArgumentParser(raw_args)
    parser.add_argument("--path", type=str, help="folder where are stored the models.", default='.')
    parser.add_argument("--n_ensemble", type=int, default=1, help='number of models to select for testing (ensembling if > 1). better to choose odd number')
    args = parser.parse_args(raw_args)
    models = glob(os.path.join(args.path, '**/*_best.pt.tar'), recursive=True)
    rows = []
    for m in models:
        try:
            state = torch.load(m, map_location='cpu')
        except:
            continue
        args_m = state['args']
        references = extract_references(args_m)
        metrics = state['best_metrics']
        #metrics = convert_flatten(metrics) I flattened the metrics directly in the models.py file
        references.update(metrics)
        rows.append(references)

    ref_metric = args_m.ref_metric # extract the reference from one of the models args (last one)
    sgn_metric = args_m.sgn_metric
    df = pd.DataFrame(rows)
    df_mean_r, df_mean_rt = mean_dataframe(df)
    models_params = select_best_repeat(df=df, sgn_metric=sgn_metric, ref_metric=ref_metric, path=args.path, n_best=args.n_ensemble)
    for param in models_params:
        copy_best_to_root(args.path, param)
    df.to_csv('all_results.csv', index=False)
    df_mean_r.to_csv('mean_over_repeats.csv', index=False)
    df_mean_rt.to_csv('mean_over_tests.csv', index=False)

if __name__ == '__main__':
    main()
