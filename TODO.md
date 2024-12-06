# TODO

- [ ] Link prediction framework.
    - [ ] Sample edges for removal.
    - [ ] Keep removed edges for testing (positive samples).
    - [ ] Sample random pairs that are not connected (negative samples).
    - [ ] What strategy we use for making edge embeddings out of nodes.

Paper grid searches over $p,q \in \{0.25, 0.50, 1, 2,4\}$ which means $5\times5=25$ embedding generations per dataset.

#### BlogCatalog 10k (Jelle laptop)

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