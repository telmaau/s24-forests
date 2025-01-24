# s24-forests
Topic modeling forest discussions on Suomi24.

## Workflow

1. Collect all posts that contain a lemma from the [search word](metsäsanat_v2.xlsx) list.
- [code/compileCorpus_csv.py](./code/compileCorpus_csv.py) creates a conll and a csv file from the posts with metadata and lemmatized text.
    - There are two almost identical versions of the csv file, and I'm not sure why.. I merged them as shown in the data_prep.ipynb, and used the resulting dataset in topic model training.
- the collected corpus can be found here: (add link)

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


4. Model evaluation
- The topic keywords were human-annotated as (good/satisfactory/unsatisfactory) for coherence.
- The models are saved with pickle and safetensors. (See differences in https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html). Safetensors is lighter but does not save the submodels (umap, kmeans etc., so if you want to fine-tune the models further, picklemodel is recommended. To just explore the models, both version work just great.
- See an example notebook of model evaluation: [code/bertopic_finnish_umap_kmeans_analysis_clean.ipynb](./´code/bertopic_finnish_umap_kmeans_analysis_clean.ipynb)

## Research
This repository is part of [SeedLING](https://sites.utu.fi/seedling/) project.
