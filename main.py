import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networkx as nx

import random
from tqdm import trange, tqdm
from typing import List, Tuple, Dict
from collections import deque

import argparse


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(SkipGram, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        
    def forward(self, u: torch.Tensor, v: torch.Tensor):
        return (self.emb(u) * self.emb(v)).sum(dim=-1)
    
def train(
        args: argparse.Namespace, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: optim.lr_scheduler._LRScheduler,
        walks: torch.Tensor,
        epoch: int, 
        report: List[Dict], 
        device: torch.device
    ) -> None:
    model.train()
    model = model.to(device)
    
    # Move walks to device
    walks = walks.to(device)

    n = walks.shape[0]
    walks_per_node = 10
    window_size = 10

    idx = torch.arange(walks.shape[-1], device=device).view(-1, 1)
    off = torch.arange(-window_size, window_size + 1, device=device)
    off = off[off != 0]  # [..., -2, -1, 1, 2, ...]

    window_idx = idx + off  # (len(walk), 2*window_size)
    mask = (window_idx >= 0) & (window_idx < walks.shape[-1])  # Legal indices of the walk

    neighbors_idx = window_idx[mask].view(-1)
    centers_idx = idx.repeat(1, 2*window_size)[mask].view(-1)
    pairs_idx = torch.stack([centers_idx, neighbors_idx], dim=1)

    # freq = torch.bincount(walks.flatten()).type(torch.float32).to(device)**(3/4)
    
    steps = 0
    total_loss = []
    for j in tqdm(torch.randperm(len(walks)), disable=(not args.verbose), desc=f'epoch:[{epoch:02d}/{args.epochs:02d}]'):
        optimizer.zero_grad()

        u, v = walks[j][:,pairs_idx].view(-1, 2).T

        # Move tensors to device
        u = u.to(device)
        v = v.to(device)

        # Negative samples
        u_neg = u.repeat_interleave(args.negative_samples)
        w_neg = torch.randint(0, len(walks), (len(u_neg),), device=device)
        # w_neg = torch.multinomial(freq, len(u_neg), replacement=True)

        loss = -(F.logsigmoid(model(u, v)).mean() + F.logsigmoid(-model(u_neg, w_neg)).mean())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        steps += 1
        total_loss.append(loss.item()/10)

    plt.semilogy(total_loss)
    plt.savefig(f'./loss-curve.png')
    plt.close()

    
def eval(args: argparse.Namespace, model: nn.Module, G: nx.Graph) -> List[float] :
    model.eval()

    with torch.no_grad() : X = model.emb.weight.detach().cpu().numpy()

    # get the number of groups
    with open('./datasets/BlogCatalog/groups.csv', 'r') as handle : 
        n_groups = len(handle.readlines())

    # setup the multi-label target `y`
    y = np.zeros((len(G), n_groups), dtype=np.int32)
    with open('./datasets/BlogCatalog/group-edges.csv', 'r') as handle : 
        line = handle.readline()
        while line != '':
            user, group = map(int, line.strip().split(','))
            y[user-1, group-1] = 1

    
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score

    clf = OneVsRestClassifier(LogisticRegression(penalty='l2'), n_jobs=-1)

    kf = KFold(n_splits=args.folds, shuffle=True)
    f1 = []
    for train_index, test_index in kf.split(X) :
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        f1.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        if (args.verbose) : print(f1[-1])
    return f1
    

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(description='node2vec')

    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--negative-samples', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--alpha-min', type=float, default=0.0005)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=1)

    return parser.parse_args()

def main(args: argparse.Namespace) -> List[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.verbose) : print('loading graph...')
    G = nx.read_edgelist('./datasets/BlogCatalog/edges.csv', delimiter=',', nodetype=int)

    n_walks = 10
    n_steps = 80

    if (args.verbose) : print('loading walks...')
    walks = torch.empty((G.number_of_nodes()*n_walks, n_steps+1), dtype=torch.int32)
    with open('./datasets/BlogCatalog/paths.txt', 'r') as handle :
        for i, line in enumerate(handle.readlines()) :
            walks[i] = torch.tensor(list(map(int, line.strip().split())))
    walks -= 1
    walks = walks.view(G.number_of_nodes(), n_walks, n_steps+1)

    model = SkipGram(len(G.nodes), args.emb_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.alpha)
    lambda_lr = lambda epoch: 1. - (epoch / (args.epochs * len(walks))) * (1. - args.alpha_min / args.alpha)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    report = []

    for epoch in range(args.epochs) : 
        train(args, model, optimizer, scheduler, walks, epoch+1, report, device)

    return eval(args, model, G)

if (__name__ == '__main__') : main(parse_args())
