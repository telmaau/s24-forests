# script to train bertopic models
from nltk.corpus import stopwords
from bertopic import BERTopic
import pandas as pd
import sys
from umap import UMAP
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# data


fpath = sys.argv[1]

# embeddings
embedding_model = sys.argv[2]

# dimension reduction model
dim_model_name = sys.argv[3]

# clustering model
cluster_model_name = sys.argv[4]



# set random seed
import numpy as np
import random

my_seed=99
random.seed(my_seed)
np.random.seed(my_seed)


# sentence transformer

from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('TurkuNLP/sbert-cased-finnish-paraphrase')


#vectorizer_model = CountVectorizer(stop_words=stop, min_df=5, max_df=0.95)
#ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

import hdbscan
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,        # Minimum cluster size
    min_samples=10,              # Minimum samples to be a core point
    metric='euclidean',            # Distance metric
    cluster_selection_method='eom',  # Cluster selection method
    prediction_data=True        # Enable prediction for unseen data
)


k_model = KMeans(n_clusters=150, max_iter = 300, n_init=30, random_state=my_seed)
pca_model = PCA(n_components=5, random_state=my_seed)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=my_seed)

# embeddings

import h5py

# load in embeddings

if embedding_model == 'TurkuNLP/sbert-cased-finnish-paraphrase':
    epath="/scratch/project_2008526/telmap/suomi24/new_embeddings.h5"
    if not os.path.isfile(epath):
        
        print("creating embeddings")
        em= 'TurkuNLP/sbert-cased-finnish-paraphrase'
        model = SentenceTransformer(em)
        embeddings = model.encode(documents, convert_to_numpy=True)
        with h5py.File(epath, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip")
    else:
        print(f"Loading embeddings '{epath}'. \n")
        with h5py.File(epath, "r") as f:
            embeddings = f["embeddings"][:]
            
elif embedding_model == 'sentence-transformers/paraphrase-xlm-r-multilingual-v1':
    epath="/scratch/project_2008526/telmap/suomi24/xlm_train_embeddings.h5"
    if not os.path.isfile(epath):
        print("Creating embeddings")
        em= 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
        model = SentenceTransformer(em)
        embeddings = model.encode(documents, convert_to_numpy=True)
        with h5py.File(epath, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip")
    else:
        print(f"Loading embeddings '{epath}'. \n")
        with h5py.File(epath, "r") as f:
            embeddings = f["embeddings"][:]
    

if dim_model_name == "pca":
    dim_model = pca_model
else:
    dim_model = umap_model
    
if cluster_model_name == "kmeans":
    cluster_model=k_model
else:
    cluster_model=hdbscan_model


# estimate a good nr of k with umap
from sklearn.metrics import silhouette_score

wcss = []
k_values = [50,75,100,125,150,175,200,225,250,275,300]
silhouette_scores = []

X = dim_model.fit_transform(embeddings)

for k in k_values:
    kmeans = KMeans(n_clusters=k, max_iter = 300, n_init=30, random_state=my_seed)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # WCSS is the inertia attribute in sklearn's KMeans
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)
    print(kmeans.inertia_, score, "\n")


import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2, figsize=(9,4))
axs[0].plot(k_values, wcss, 'bx-')
axs[1].plot(k_values,silhouette_scores, 'bx-')

axs[0].set_title('Elbow Method for Optimal k')
axs[0].set_xlabel('Number of Clusters (k)')
axs[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
axs[1].set_title('Silhouette score for Optimal k')
axs[1].set_xlabel('Number of Clusters (k)')
axs[1].set_ylabel("silhouette score")


fig.show()

print("Done!\n")
