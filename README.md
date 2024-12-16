# Feature Smoothing Node2Vec
In Node2Vec numerical vector representations (embeddings) are learned
by using random walks on the graph. Doing this results in embeddings
that capture numerical features of the strucutre of the graph and 
node's pressence within communities such that they can be used in 
machine learning tasks. 

To test these embeddings we perform multi-label classification and 
link prediction based on their embeddings.

We propose an updating rule for diffusing information to neighbors
by doing a smooth update of the embeddings based on the embeddings of 
the neighbors. That is, we add a small amount of embeddings of our
neighbors to our own embedding. We hypothesize this will lead to 
more expressive embeddings since random walks rarely visit all 
direct neigbors.

# Files
```
snacs-project/
    datasets/                       Raw datasets that are prepared in `prepare.py`
    node2vec/                       Forked Node2Vec source code; check readme in there
    results/                        Results of the node classification grid search
    analysis.ipynb                  Analysis of node classification grid search
    demo.ipynb                      Demo notebook as used in the presentation to explain Node2Vec
    link-prediction.py              Classification of links
    node-classification.py          Classification of nodes
    prepare.py                      Preparation of raw datasets to be used in `./node2vec/`
    utils.py                        Utility functions for loading/preparing embeddings 
```

# How to use
The preparation of the datasets needs to be ran only once.
This prepares the raw datasets (from different formats) to 
the expected format of the original `./node2vec/` implementation.

After preparation of the datasets embeddings can be generated using
the original Node2Vec source code. Note a couple of adjustments
have been made to make the code compatible in Python 3. See the
readme in `./node2vec/` to see how to run the training of the 
embeddings.

After training embeddings for required (p,q) combinations
analysis can be done for smoothing embeddings for the node classification
task using `node-classification.py` and `link-predcition.py` for
link prediction.

# Installation

```
[python=3.10]
pip install numpy matplotlib scipy scikit-learn gensim
```