import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from typing import Dict, List

def read_file(
    path: str,
    id2idx: Dict[str, int],
    dtype: type=float
) -> List[List[float | int]] :
    
    with open(path, 'r') as handle:
        (n, d) = map(int, handle.readline().strip().split())

        data = np.empty((n, d), dtype=dtype)
        while (line := handle.readline().strip()):
            (id, *data_field) = line.split()
            data[id2idx[id]] = list(map(float, data_field))

    return data


def load_embeddings(path: str, id2idx: Dict[str, int]) -> List[List[float]] : 
    return read_file(path, id2idx, dtype=float)


def load_labels(path: str, id2idx: Dict[str, int]) -> List[List[int]] : 
    return read_file(path, id2idx, dtype=int)


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


def embeddings_smoothing(
    G: nx.Graph, 
    id2idx: Dict[str, int],
    X: List[List[float]], 
    gamma=0.05
) -> List[List[float]] :
    # diffusing the learned embeddings based on their neighbors
    deg_avg = sum(d for (n, d) in G.degree()) / G.number_of_nodes()

    Xprime = X.copy()
    for u in G.nodes() : 
        for v in G.neighbors(u) : 
            Xprime[id2idx[u]] += gamma*(G.degree(v)/deg_avg)*X[id2idx[v]]
    return Xprime