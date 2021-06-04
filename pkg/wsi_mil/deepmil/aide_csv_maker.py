import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold

def random_sampling(t, N, replace=False):
    if N is None:
        N = len(t)
    indices = np.random.choice(t.index, N, replace=replace)
    return t.loc[indices]

def select(df, dictio, N=None):
    bools = np.ones(len(df), dtype=bool)
    for col in dictio:
        if type(dictio[col]) == list:
            tmp = []
            for v in dictio[col]:
                tmp.append(np.logical_and(bools, df[col] == v))
            bools = reduce(np.logical_or, tmp)
        else:
            bools = np.logical_and(bools, df[col] == dictio[col])
    selection = random_sampling(df[bools], N)
    return selection, len(selection)

def make_new_var(list_vars): 
    def get_func(x):
        new = ''
        for v in list_vars:
            if v is not None:
                new += str(x[v]) + '_'
        return new
    return get_func

class DatasetSampler:
    def __init__(self, table, target, k=5):
        """
        equ_vars : nom des variables devant etre équilibré wrt target.
        fix_vars : dictio {variable : fix_value} pour fixer des variables.
        target : target for classification, used for balancong the dataset
        """
        self.table = pd.read_csv(table)
        self.target = target
        self.k = k

    def set_test(self, table, equ_vars):
        """
        divide into k-fold the dataset (creation of test_set)
        according to all the variables in equ_vars and target.
        """
        if type(equ_vars) != list:
            equ_vars = [equ_vars]
        equ_vars = equ_vars + [self.target]
        new_var = self.table.apply(make_new_var(equ_vars), axis=1)
        table['stratif'] = new_var
        skf = StratifiedKFold(n_splits=self.k, shuffle=True)
        y = table['stratif'].values
        X = list(range(len(y)))
        dic_test = dict()
        test_vec = np.zeros(len(y))
        for o, (train, test) in enumerate(skf.split(X, y)):
            for i in test:
                test_vec[i] = o
        table['test'] = test_vec
        return table

    def make_dataset(self, equ_vars:list, upsample=True,fix_vars=None):
        """make_dataset.

        returns a table with a stratified and balanced dataset.

        :param equ_vars: list of variable to balance amond the classes of the 
        target variable.
        :type equ_vars: list
        :param upsample: strategy to sample the minority/majority class
        for balancing the dataset.
        :param fix_vars: dictionary setting the variables that you want to be fixed.
        """
        table = select(self.table, dictio=fix_vars)[0]
        table = self.set_test(table, equ_vars)
        new_var = self.table.apply(make_new_var(equ_vars), axis=1)
        table['equ'] = new_var
        lv_values = set(new_var.values)
        tmp = []
        for lv in lv_values:
            tmp_table = []
            tmp_N = []
            for tv in self.target_values:
                d = {'equ':lv, self.target:tv}
                sub_table = select(table, d)
                tmp_table.append(sub_table[0])
                tmp_N.append(sub_table[1])
            sup = max(tmp_N) if upsample else min(tmp_N)
            for tt in tmp_table:
                if len(tt) == sup:
                    tmp.append(tt)
                else:
                    tmp.append(random_sampling(tt, sup, replace=upsample))
        selection = pd.concat(tmp)
        return selection
 
    def downsample(self, fix_vars:dict, equ_vars:list):
        """
        Not used (part of make_dataset_now) keep until sure make_dataset works.
        """
        table = select(self.table, dictio=fix_vars)[0]
        self.target_values = set(table[self.target].values)
        new_var = table.apply(make_new_var(equ_vars), axis=1)
        table['equ'] = new_var
        lv_values = set(new_var.values)
        tmp = []
        for lv in lv_values:
            tmp_table = []
            tmp_N = []
            for tv in self.target_values:
                d = {'equ':lv, self.target:tv}
                sub_table = select(table, d)
                tmp_table.append(sub_table[0])
                tmp_N.append(sub_table[1])
            smallest = min(tmp_N)
            for tt in tmp_table:
                tmp.append(random_sampling(tt, smallest, replace=False))
        selection = pd.concat(tmp)
        selection = self.set_test(selection, equ_vars)
        return selection

def test_stratif(table, equ_vars, target, k):
    """
    divide into k-fold the dataset (creation of test_set)
    according to all the variables in equ_vars and target.
    """
    table = pd.read_csv(table)
    if type(equ_vars) != list:
        equ_vars = [equ_vars]
    equ_vars = equ_vars + [target]
    new_var = table.apply(make_new_var(equ_vars), axis=1)
    table['stratif'] = new_var
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    y = table['stratif'].values
    X = list(range(len(y)))
    dic_test = dict()
    test_vec = np.zeros(len(y))
    for o, (train, test) in enumerate(skf.split(X, y)):
        for i in test:
            test_vec[i] = o
    table['test'] = test_vec
    return table





        


