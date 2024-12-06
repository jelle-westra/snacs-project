# Diffusing Node2Vec
In Node2Vec numerical vector representations (embeddings) are learned
by using random walks on the graph. Doing this results in embeddings
that capture numerical features of the strucutre of the graph and 
node's pressence within communities such that they can be used in 
machine learning tasks. 

To test these embeddings we perform multi-label classification on 
purely based on their embeddings.

We propose an updating rule for diffusing information to neighbors
by doing a smooth update of the embeddings based on the embeddings of 
the neighbors. That is, we add a small amount of embeddings of our
neighbors to our own embedding. We hypothesize this will lead to 
more expressive embeddings since random walks rarely visit all 
direct neigbors.

# Files
```
snacs-project/
    datasets/
        BlogCatalog/
            edges.csv               Edgelist: (User1, User2)
            group-edges.csv         Classes: (User, Group)
            groups.csv              Class indices
            nodes.csv               Node indices
        PPI/
            Homo_sapiens.mat        MATLAB data file containing: Adjacency Matrix, and node classes
    node2vec/                       Node2Vec source code; check readme in there
    BlogCatalog.ipynb               Analysis of the BlogCatalog dataset
    PPI.ipynb                       Analysis of the PPI dataset
    demo.ipynb                      Demo notebook as used in the presentation to explain Node2Vec
```

# Installation

```
[python=3.10]
pip install numpy matplotlib scipy scikit-learn gensim
```