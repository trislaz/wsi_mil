from sklearn.model_selection import StratifiedKFold
import pandas as pd
from argparse import ArgumentParser
from deepmil.aide_csv_maker import test_stratif

# From a csv table, creates an other csv table with a "test" column, stating in which test fold are each line.
# The created Kfold is stratified with respect to $target_name, a column of the csv.

parser = ArgumentParser()
parser.add_argument('--table', type=str)
parser.add_argument('--target_name', type=str)
parser.add_argument('--equ_vars', type=str, default=None, help='variables to keep in stratif vars')
parser.add_argument('-k', type=int, help='number of folds')
args = parser.parse_args()

#table = pd.read_csv(args.table_in)
target = args.target_name
equ_vars = args.equ_vars
if equ_vars is not None:
    equ_vars = args.equ_vars.split(',')

table = test_stratif(args.table, equ_vars, args.target_name, args.k)
table.to_csv(args.table, index=False)






