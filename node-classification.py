import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import os
import argparse
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Any

from prepare import prepare
from utils import load_embeddings, load_labels, embeddings_smoothing

# thanks to hermidalc, https://github.com/scikit-learn/scikit-learn/issues/12939
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def cross_validate_f1(
    X: List[List[float]], 
    y: List[List[int]], 
    folds: int=10, 
    verbose: bool=True
) -> List[float] :
    # macro-f1 cv on logistic 1-vs-rest logistic regression
    
    clf = OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000), n_jobs=-1)
    kf = KFold(n_splits=folds, shuffle=True)
    f1 = []
    for j, (train_index, test_index) in enumerate(kf.split(X)) :
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        if (verbose) : print(f'[{j+1:02d}/{folds:02d}] {f1[-1]:.5f}')
    if (verbose) : print(f'mean: {np.mean(f1):.5f}')
    return np.array(f1)


def parse_args():
    parser = argparse.ArgumentParser(description='Node Classification')
    parser.add_argument('--dataset', type=str, default='PPI')
    parser.add_argument('--gamma-min-order', type=int, default=-6)
    parser.add_argument('--gamma-max-order', type=int, default=1)
    parser.add_argument('--gamma-samples', type=int, default=22)
    parser.add_argument('--folds', type=int, default=10)

    return parser.parse_args()


if (__name__ == '__main__') : 
    args = parse_args()

    print('Preparing the dataset...')
    prepare(args.dataset)

    G = nx.read_edgelist(f'./node2vec/graph/{args.dataset}.edgelist')
    id2idx = {id: idx for (idx, id) in enumerate(G.nodes())}

    paths = glob(f'./node2vec/emb/{args.dataset}/*.emb')

    gammas = np.logspace(args.gamma_min_order, args.gamma_max_order, args.gamma_samples)

    y = load_labels(f'./node2vec/label/{args.dataset}.lab', id2idx)

    print('Evaluating...')
    for path in paths :
        (p, q) = (float(s[1:]) for s in os.path.basename(os.path.normpath(path))[:-4].split('_'))
        pbar = tqdm(total=len(gammas) + 1, leave=False, desc=f'[p={p:.2f},q={q:.2f}]')

        X = load_embeddings(paths[0], id2idx)
        f1 = cross_validate_f1(X, y, folds=args.folds, verbose=False)
        pbar.update(1)

        f1_diffused = np.zeros((len(gammas), args.folds), dtype=np.float32)
        for (j, gamma) in enumerate(gammas) :
            Xprime = embeddings_smoothing(G, id2idx, X, gamma)
            f1_diffused[j] = cross_validate_f1(Xprime, y, folds=args.folds, verbose=False)
            pbar.update(1)
        pbar.close()

        diffused_best = f1_diffused.mean(axis=1).argmax()

        print(f'[p={p:.2f},q={q:.2f}]: base {f1.mean():.4f} | best_gamma {gammas[diffused_best]} | best_diffused {f1_diffused[diffused_best].mean():.4f}')
    