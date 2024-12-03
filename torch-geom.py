import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

from tqdm import tqdm
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--diffuse', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.01)

    return parser.parse_args()

def main():
    args = parse_args()

    G = nx.read_edgelist('./datasets/BlogCatalog/edges.csv', delimiter=',', nodetype=int)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    edges = np.empty((2*G.number_of_edges(), 2), dtype=np.int32)
    for i, (u,v) in enumerate(G.to_directed().edges()) : edges[i] = (u, v)
    edges -= 1 

    model = Node2Vec(
        torch.tensor(edges.T, dtype=torch.int64, device=device),
        embedding_dim=128, 
        walk_length=80, 
        context_size=10, 
        walks_per_node=10, 
        num_negative_samples=5,
        p=1,
        q=1,
        num_nodes=G.number_of_nodes()
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=0)

    # train
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, leave=False):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'[{epoch+1}/{args.epochs}] {total_loss:.2f}')

    # eval
    X = model.embedding.weight.data.detach().cpu().numpy()

    # the diffusion trick
    if (args.diffuse) :
        deg_avg = sum(d for (n, d) in G.degree()) / G.number_of_nodes()

        Xprime = X.copy()
        for u in G.nodes() : 
            for v in G.neighbors(u) : 
                Xprime[u-1] += args.gamma*(G.degree(v)/deg_avg)*X[v-1]

        X = Xprime.copy()

    # get the number of groups
    with open('./datasets/BlogCatalog/groups.csv', 'r') as handle : 
        n_groups = len(handle.readlines())

    # setup the multi-label target `y`
    y = np.zeros((len(G), n_groups), dtype=np.int32)
    with open('./datasets/BlogCatalog/group-edges.csv', 'r') as handle : 
        while ((line := handle.readline()) != '') :
            user, group = map(int, line.strip().split(','))
            y[user-1, group-1] = 1

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000), n_jobs=-1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(args.lr, args.batch_size, f1)

if (__name__ == '__main__') : main()