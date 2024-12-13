import numpy as np
import networkx as nx
from scipy.io import loadmat

import os
from typing import List, Dict


def prepare_BlogCatalog() -> None :
    dataset = 'BlogCatalog'

    # exporting edgelist
    G = nx.read_edgelist(f'./datasets/{dataset}/edges.csv', delimiter=',')
    nx.write_edgelist(G, f'./node2vec/graph/{dataset}.edgelist')

    G = nx.read_edgelist(f'./node2vec/graph/{dataset}.edgelist')
    id2idx = {id: idx for (idx, id) in enumerate(G.nodes())}

    # number of labels
    with open('./datasets/BlogCatalog/groups.csv', 'r') as handle : 
        n_groups = len(handle.readlines())

    group2idx = {str(group+1): group for group in range(n_groups)}

    # setup the multi-label target `y`
    y = np.zeros((len(G), n_groups), dtype=np.int32)
    with open('./datasets/BlogCatalog/group-edges.csv', 'r') as handle : 
        while ((line := handle.readline()) != '') :
            (id, group) = line.strip().split(',')
            y[id2idx[id], group2idx[group]] = 1

    # exporting labels
    with open(f'./node2vec/label/{dataset}.lab', 'w') as handle:
        handle.write(f'{len(G)} {y.shape[1]}\n')
        for node in G.nodes():
            handle.write(node + ' ' + ' '.join(map(str, y[id2idx[node]])) + ' \n')


def generate_labels_PPI() -> List[List[int]] :
    PPI_data = loadmat('./datasets/PPI/Homo_sapiens.mat')

def generate_labels_Wikipedia() -> List[List[int]] : pass

if (__name__ == '__main__') :
    if not(os.path.exists('./node2vec/label')) : os.mkdir('./node2vec/label')

    prepare_BlogCatalog()
    # generate_labels_PPI()
    # generate_labels_Wikipedia()