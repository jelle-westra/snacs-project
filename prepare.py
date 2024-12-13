import numpy as np
import networkx as nx
from scipy.io import loadmat

import os
import random
from typing import Tuple, Set, Any, List, Dict

def prepare(dataset: str, link_prediction: bool = False, edgelist_path: str = "", labels_path: str = "") -> None :
    match dataset:
        case 'BlogCatalog':
            prepare_csv('BlogCatalog', './datasets/BlogCatalog/edges.csv', './datasets/BlogCatalog/groups.csv', './datasets/BlogCatalog/group-edges.csv')	
        case 'PPI':
            prepare_mat('PPI', './datasets/PPI/Homo_sapiens.mat', 'network', 'group')
        case 'Wikipedia':
            prepare_mat('Wikipedia', './datasets/wikipedia/wikipedia.mat', 'network', 'group')
        case 'facebook':
            prepare_txt('facebook', './datasets/facebook/facebook_combined.txt', ' ')
        case 'ca-AstroPh':
            prepare_txt('ca-AstroPh', './datasets/ca-AstroPh/ca-AstroPh.txt', '\t')
        case _:
            raise ValueError(f'Invalid dataset: {dataset}') 
    if link_prediction:
        prepare_link_prediction(dataset)

def prepare_link_prediction(dataset: str):
    G = nx.read_edgelist(f'./node2vec/graph/{dataset}.edgelist', nodetype=int)
    G_giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n = G_giant.number_of_edges()//2
    
    removed = random_remove_edges(G_giant, n)
    nx.write_edgelist(G_giant, f'./node2vec/graph/{dataset}-half.edgelist')
    with open(f'./node2vec/graph/{dataset}-removed.edgelist', 'w') as handle: 
        handle.writelines(f'{u} {v}\n' for (u, v) in removed)

def prepare_txt(dataset: str, edgelist_path: str, delimiter: str = ',') -> None :
    try:
        valid_file(edgelist_path, '.txt')
    except Exception as e:
        print(e)
        return
    
    G = nx.read_edgelist(edgelist_path, delimiter=delimiter, nodetype=int)
    nx.write_edgelist(G, f'./node2vec/graph/{dataset}.edgelist')

def prepare_csv(dataset: str, edgelist_path: str, labels_path: str, edge_labels_path: str, delimiter: str = ',') -> None :
    try:
        valid_file(edgelist_path, '.csv')
        valid_file(labels_path, '.csv')
        valid_file(edge_labels_path, '.csv')
    except Exception as e:
        print(e)
        return

    # exporting edgelist
    G = nx.read_edgelist(edgelist_path, delimiter=delimiter)
    nx.write_edgelist(G, f'./node2vec/graph/{dataset}.edgelist')
    id2idx = {id: idx for (idx, id) in enumerate(G.nodes())}

    # number of labels
    with open(labels_path, 'r') as handle : 
        n_labels = len(handle.readlines())

    group2idx = {str(label+1): label for label in range(n_labels)}

    # setup the multi-label target `y`
    y = np.zeros((len(G), n_labels), dtype=np.int32)
    with open(edge_labels_path, 'r') as handle : 
        while ((line := handle.readline()) != '') :
            (id, group) = line.strip().split(',')
            y[id2idx[id], group2idx[group]] = 1

    # exporting labels
    with open(f'./node2vec/label/{dataset}.lab', 'w') as handle:
        handle.write(f'{len(G)} {y.shape[1]}\n')
        for node in G.nodes():
            handle.write(node + ' ' + ' '.join(map(str, y[id2idx[node]])) + ' \n')

def prepare_mat(dataset: str, mat_path: str, network: str, labels: str) -> None :
    try:
        valid_file(mat_path, '.mat')
    except Exception as e:
        print(e)
        return
    
    mat = loadmat(mat_path)
    G = nx.from_scipy_sparse_array(mat[network])
    nx.write_edgelist(G, f'./node2vec/graph/{dataset}.edgelist')

    y = mat[labels].toarray().astype(int)
    id2idx = {id: idx for (idx, id) in enumerate(G.nodes())}
    with open(f'./node2vec/label/{dataset}.lab', 'w') as handle:
        handle.write(f'{len(G)} {y.shape[1]}\n')
        for node in G.nodes():
            handle.write(str(node) + ' ' + ' '.join(map(str, y[id2idx[node]])) + ' \n')
    
def valid_file(path: str, fileType: str):
    if not os.path.isfile(path):
        raise Exception(f"{path} file does not exist")
    if not path.endswith(fileType):
        raise Exception(f"{path} not of typ {fileType}")

Node = Any

def random_remove_edges(G_giant: nx.Graph, n: int) -> Set[Tuple[Node, Node]] :
    G = G_giant.copy()
    # inplace removal of random edges s.t. G stays connected
    edges = list(G_giant.edges())
    removed = set()
    i = 0
    while (i < n):
        (idx,) = random.sample(range(len(edges)), 1)
        (u, v) = edges.pop(idx)
        if not(G.degree(u) == 1) and not(G.degree(v) == 1) :
            G_giant.remove_edge(u, v)
            i += 1
            removed.add((u, v))
    return removed

if (__name__ == '__main__') :
    if not(os.path.exists('./node2vec/label')) : os.mkdir('./node2vec/label')
    # Classification 
    prepare('BlogCatalog')
    prepare('PPI')
    prepare('Wikipedia')
    # Link Prediction
    prepare('facebook', link_prediction=True)
    prepare('ca-AstroPh', link_prediction=True)