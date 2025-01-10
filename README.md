# s24-forests
Topic modeling forest discussions in Suomi24

## Workflow

1. Collect all posts that contain a lemma from the search word list.
- the collected corpus can be found here:

2. Estimate the number of topics for topic models using K-means.

3. Train topic models with different combinations.

| Embedding model              | Dimensionality reduction | Clustering |
| :---------------- | :------: | ----: |
| Finnish       |   UMAP  | HDBSCAN |
| Finnish       |   UMAP  | K-means |
| Finnish       |   PCA  | HDBSCAN |
| Finnish       |   PCA  | K-means |
| Multilingual      |   UMAP  | HDBSCAN |
| Multilingual       |   UMAP  | K-means |
| Multilingual       |   PCA  | HDBSCAN |
| Multilingual       |   PCA  | K-means |


4. add something
