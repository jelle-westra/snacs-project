from utils import load_embeddings, embeddings_smoothing
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

def data_pipeline(
    G: nx.Graph,
    edges: Set[Tuple[int, int]], 
    id2idx: Dict[str, int], 
    embeddings: List[List[float]]
) -> Tuple[List[List[float]], List[int]] :
    
    m = len(edges)
    X = np.empty((2*m, embeddings.shape[1]), dtype=np.float32)
    y = np.zeros(len(X), dtype=np.float32)
    y[:m] = 1

    # positive examples
    for i, (u, v) in enumerate(edges):
        X[i] = embeddings[id2idx[u]] * embeddings[id2idx[v]]
    # negative examples
    i = 0
    nodes = list(G.nodes())
    while (i < m) :
        (u, v) = random.sample(nodes, 2)
        if not(G.has_edge(u, v)) :
            X[m+i] = embeddings[id2idx[u]] * embeddings[id2idx[v]]
            i += 1

    return (X, y)

def evaluate(G_half: nx.Graph, removed: Set[Tuple[int, int]], id2idx: Dict[str, int], embeddings: List[List[float]]) -> float :
    (X_train, y_train) = data_pipeline(G_half, set(G_half.edges()), id2idx, embeddings)
    (X_test, y_test) = data_pipeline(G_half, removed, id2idx, embeddings)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return roc_auc_score(y_test, y_pred)


if (__name__ == '__main__') :
    dataset = 'ca-AstroPh'
    folds = 10

    (p,q) = (1, 1)
    
    # note, G_half still contains all the nodes
    G_half = nx.read_edgelist(f'./node2vec/graph/{dataset}-half.edgelist')
    id2idx = {id : idx for (idx, id) in enumerate(sorted(G_half.nodes()))}

    with open(f'./node2vec/graph/{dataset}-removed.edgelist', 'r') as handle:
        removed = {tuple(line.strip().split()) for line in handle}

    embeddings = load_embeddings(f'./node2vec/emb/{dataset}-half.emb', id2idx)

    auc_base = evaluate(G_half, removed, id2idx, embeddings)

    gammas = np.logspace(-6, 1, 22)
    auc_smoothed = np.zeros(len(gammas), dtype=np.float32)
    for (j, gamma) in tqdm(enumerate(gammas), total=len(gammas)) :
        embeddings_smooth = embeddings_smoothing(G_half, id2idx, embeddings, gamma)
        auc_smoothed[j] = evaluate(G_half, removed, id2idx, embeddings_smooth)

    smoothed_best = auc_smoothed.argmax()
    print(f'[p={p:.2f},q={q:.2f}]: base {auc_base:.4f} | best_gamma {gammas[smoothed_best]} | best_smoothed : {auc_smoothed[smoothed_best].mean():.4f}')