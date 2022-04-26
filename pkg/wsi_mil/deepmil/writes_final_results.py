"""
Horrible le systeme de soft voting,  refaire entièrement quand fini
"""
from .test import main as main_test
from .predict import load_model
from argparse import ArgumentParser
from functools import reduce
from sklearn import metrics
import numpy as np
from glob import glob
import os
import pandas as pd

def extract_test_repeat(path):
    path = os.path.basename(path).split('.')[0]
    test, repeat = [int(s) for s in path.split('_') if s.isdigit()]
    return test, repeat

def assert_identity(i1, i2):
    """
    asserts that all indices are in the same sequence in the res list.
    """
    assert list(i1) == list(i2), "the sequence of images are different between several models"
    return i2

def soft_voting(final_res, table):
    """soft_voting.

    Soft ensembling of the models related to a same test set.
    :param final_res: dict, dictionnary with all the final results dictionnary 
    corresponding each to a tested model.
    :param table: master table.
    :return dict, dict [performance dictionnary, prediction for each WSI] 
    """
    proba_preds = []
    preds = []
    ids = []
    tests = []
    df_res = []
    for test in final_res.keys():
        res = final_res[test]
        indices = reduce(assert_identity, [x['ids'] for x in res])
        gt = reduce(assert_identity, [x['gt'] for x in res])
        scores = [x['scores'] for x in res]
        scores = np.stack(scores) # logsoftmaxed(logits)
        voting_scores = scores.mean(0)
        elected_pred = list(np.argmax(voting_scores, axis=1))
        elected_proba = list(np.max(voting_scores, axis=1))

        # to compute result_table
        proba_preds += elected_proba
        preds += [res[0]['label_encoder'].inverse_transform([x])[0] for x in elected_pred]
        ids += indices
        tests += [test] * len(indices)

        metrics_dict = compute_metrics(gt, elected_pred, voting_scores, num_class=voting_scores.shape[-1])
        metrics_dict['test'] = test
        df_res.append(metrics_dict)

    df_res = pd.DataFrame(df_res)
    result_table = fill_table(table, proba_preds, preds, ids, tests)
    return df_res, result_table

def fill_table(table, proba_preds, preds, ids, tests):
    """fill_table.

    Fills a copy of the master table with logits and predictions.

    :param table: pd.DataFrame, master table.
    :param proba_preds: list, posterior probability of the predictions.
    :param preds: list, predictions.
    :param ids: ID of the tested slides.
    :param tests: test folds of the tested slides.
    """
    """
    """
    pi_scores = []
    pi_preds = []
    pi_tests = []
    def is_in_set(x):
        if x['ID'] in ids:
            return True
        else: 
            return False
    table['take'] = table.apply(is_in_set, axis=1)
    table = table[table['take']]
    for i in table['ID'].values:
        index = ids.index(i)
        pi_scores.append(proba_preds[index]) #pi = repermuté dans le sens de table
        pi_preds.append(preds[index])
        pi_tests.append(tests[index])
    table['proba_preds'] = pi_scores
    table['prediction'] = pi_preds
    table['test'] = pi_tests
    return table

def compute_metrics(y_true, y_pred, scores, num_class=2):
    """compute_metrics.

    Compute performances metrics.

    :param y_true: list or ndarray, labels
    :param y_pred: list or ndarray, discrete prediction
    :param scores: list or ndarray, logits
    :param num_class: number of classes.
    :return dict, key(name of metric) value(perfomance measure)
    """
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0)
    metrics_dict = {
            'accuracy': report['accuracy'] , 
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1-score": report['macro avg']['f1-score'], 
            "ba": metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            }
    if num_class <= 2:
        metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=y_true, y_score=scores[:,1])
    return metrics_dict

def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument('--path', default='.', type=str, help='path to the folder where the best models are stored.')
    args = parser.parse_args(raw_args)

    models_path = glob(os.path.join(args.path, 'model_best_test_*_repeat_*.pt.tar'), recursive=True)
    model = load_model(models_path[0], 'cpu')
    table = pd.read_csv(model.args.table_data)
    final_res = dict() 
    for model in models_path:
        res = main_test(model_path=model)
        res['test'], res['repeat'] = extract_test_repeat(model)
        if res['test'] not in final_res.keys():
            final_res[res['test']] = []
        final_res[res['test']].append(res)
    df_res, result_table = soft_voting(final_res, table=table)
    mean = df_res.mean(axis=0).to_frame().transpose()
    std = df_res.std(axis=0).to_frame().transpose()
    mean['test'] = 'mean'
    std['test'] = 'std'
    df_res = pd.concat([df_res, mean, std])
    df_res = df_res.set_index('test')
    df_res.to_csv('final_results.csv')
    result_table.to_csv('resuts_table.csv', index=False)

