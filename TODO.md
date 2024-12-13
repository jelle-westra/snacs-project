# TODO

- [ ] Link prediction framework.
    - [ ] Sample edges for removal.
    - [ ] Keep removed edges for testing (positive samples).
    - [ ] Sample random pairs that are not connected (negative samples).
    - [ ] What strategy we use for making edge embeddings out of nodes.

Paper grid searches over $p,q \in \{0.25, 0.50, 1, 2,4\}$ which means $5\times5=25$ embedding generations per dataset.

#### BlogCatalog 10k (Jelle laptop)

[p=0.25,q=2.00]: base 0.9680 | best_gamma 2.1544346900318822e-06 | best_diffused : 0.9715                              
[p=0.25,q=2.00]: base 0.9686 | best_gamma 2.1544346900318822e-06 | best_diffused : 0.9742                              
[p=0.25,q=2.00]: base 0.9697 | best_gamma 4.641588833612782e-06 | best_diffused : 0.9753                               
[p=0.25,q=2.00]: base 0.9592 | best_gamma 2.1544346900318822e-06 | best_diffused : 0.9746                              
[p=0.25,q=2.00]: base 0.9713 | best_gamma 1e-05 | best_diffused : 0.9696                                               
[p=0.25,q=2.00]: base 0.9705 | best_gamma 4.641588833612772e-05 | best_diffused : 0.9727                               
[p=0.25,q=2.00]: base 0.9673 | best_gamma 4.641588833612782e-06 | best_diffused : 0.9704

| $p$ \ $q$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---------|------|-----|---|---|---|
| 0.25    |      |     |   |   |   |
| 0.5     |      |     |   |   |   |
| 1       |      |     |   |   |   |
| 2       |      |     |   |   |   |
| 4       |      |     |   |   |   |


#### PPI 4k (Jelle laptop)

| $p$ \ $q$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---------|------|-----|---|---|---|
| 0.25    |      |     |   |   |   |
| 0.5     |      |     |   |   |   |
| 1       |      |     |   |   |   |
| 2       |      |     |   |   |   |
| 4       |      |     |   |   |   |


#### WikiPedia 5k

| $p$ \ $q$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---------|------|-----|---|---|---|
| 0.25    |      |     |   |   |   |
| 0.5     |      |     |   |   |   |
| 1       |      |     |   |   |   |
| 2       |      |     |   |   |   |
| 4       |      |     |   |   |   |

#### ArXiv 18k (link prediction)

| $p$ \ $q$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---------|------|-----|---|---|---|
| 0.25    |      |     |   |   |   |
| 0.5     |      |     |   |   |   |
| 1       |      |     |   |   |   |
| 2       |      |     |   |   |   |
| 4       |      |     |   |   |   |

#### Facebook 4k (link prediction)

| $p$ \ $q$ | 0.25 | 0.5 | 1 | 2 | 4 |
|---------|------|-----|---|---|---|
| 0.25    |      |     |   |   |   |
| 0.5     |      |     |   |   |   |
| 1       |      |     |   |   |   |
| 2       |      |     |   |   |   |
| 4       |      |     |   |   |   |