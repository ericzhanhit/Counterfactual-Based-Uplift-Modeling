"""Execute selected original definitions without importing DiCE or running scripts.

Usage: python scripts/compare_legacy.py --legacy-dir /path/to/CF_DML_240901
AST extraction retains the original function/class bodies unchanged. This is
component comparison on controlled toy data, not an end-to-end historical rerun.
"""
import argparse
import ast
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from cf_uplift.learners import ESTIMATORS, LearnerConfig
from cf_uplift.metrics.uplift import evaluate_uplift
from cf_uplift.evaluation.cross_validation import write_json

def load_definitions(path, wanted, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-dir', required=True)
    parser.add_argument('--output', default='docs/audit/legacy_comparison.json')
    args = parser.parse_args()
    namespace = dict(np=np, pd=pd, BaseEstimator=BaseEstimator, warnings=warnings,
                     type_of_target=type_of_target, check_consistent_length=check_consistent_length)
    names = ['SLearner', 'TLearner', 'XLearner', 'DMLearner', 'DRLearner']
    legacy = load_definitions(Path(args.legacy_dir)/'models.py', names, namespace)
    rng = np.random.default_rng(12)
    X = rng.normal(size=(400, 3)); t = rng.binomial(1, .5, 400)
    y = X[:, 0]+t*(2+X[:, 1])+rng.normal(0, .1, 400)
    test = rng.normal(size=(60, 3))
    reg = lambda seed=42: DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)
    cl = lambda seed=42: DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)
    models = {'SL': legacy['SLearner'](reg()), 'TL': legacy['TLearner'](reg(43), reg()),
              'XL': legacy['XLearner'](reg(43), reg(), reg(45), reg(44), cl(46)),
              'DML': legacy['DMLearner'](cl(), reg(), reg()),
              'DRL': legacy['DRLearner'](cl(), reg(), reg())}
    output = {'scope': 'Original AST definitions executed unmodified on toy data; not historical experiments', 'models': {}}
    for name, model in models.items():
        original = model.fit(X, y, t).predict(test)
        new = ESTIMATORS[name](LearnerConfig('DT', {'min_samples_leaf': 5})).fit(X, t, y).predict_uplift(test)
        output['models'][name] = {'max_abs_prediction_difference': float(np.max(np.abs(original-new))),
                                 'mean_abs_prediction_difference': float(np.mean(np.abs(original-new)))}
    metric_names = ['get_cumlift', 'get_cumgain', 'get_qini', 'auuc_score', 'qini_score']
    metrics = load_definitions(Path(args.legacy_dir)/'metrics_causalml.py', metric_names,
                               {'np': np, 'pd': pd, 'RANDOM_COL': 'Random'})
    toy = pd.DataFrame({'y': [0., 1., 1., 1.], 'w': [0, 1, 0, 1], 'score': [4., 3., 2., 1.]})
    output['metrics'] = {'new': evaluate_uplift(toy.y, toy.w, toy.score),
                         'old_auuc': float(metrics['auuc_score'](toy)['score']),
                         'old_qini': float(metrics['qini_score'](toy)['score'])}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, output)
    print(target)

if __name__ == '__main__': main()

