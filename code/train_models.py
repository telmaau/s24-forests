# script to train bertopic models
from nltk.corpus import stopwords
from bertopic import BERTopic
import pandas as pd
import sys
from umap import UMAP
import os
import numpy as np
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

# nr of topics
ntopics = int(sys.argv[5])

# stopwords
custom_stop_words = ["eli","vain", "vaan","siksi","mä","sä","tällä", "silloin", "joku","jossain","se","siellä","ihan","vielä","mikään","of","and","if","the","på","och","by","myös","jo","make","van","jokin","in","sitten"]

stop = stopwords.words("finnish")
stop.extend(custom_stop_words)


# read in train data
trainpath= "/scratch/project_2008526/telmap/suomi24/corpus/s24_metsa_actors_new_text2len_train.csv"
df = pd.read_csv(trainpath, sep="\t")
documents=list(df["text"].drop_duplicates())
print("Training on", len(documents), "documents. \n")

# and test data
testpath= "/scratch/project_2008526/telmap/suomi24/corpus/s24_metsa_actors_new_text2len_test.csv"
df2 = pd.read_csv(testpath, sep="\t")
documents2=list(df2["text"].drop_duplicates())

import re

def clean_doc(doc):
    # Pattern to match any letters separated by a dot
    pattern = r"([a-zöåäA-ZÅÄÖ]+)\.([a-zåäöA-ZÅÄÖ]+)"
    pattern2 = r"([a-zåäöA-ZÅÄÖ]+)\,([a-zöäåA-ZÅÄÖ]+)"
    # Replace with a space after the dot
    if "http" and "www" not in doc:
        doc = re.sub(pattern, r"\1. \2", doc)
        doc = re.sub(pattern2, r"\1, \2", doc)
    
    doc_clean = re.sub(r"\s+", " ", doc).strip()

    words = doc_clean.split()
    # Convert words with all uppercase letters to lowercase
    processed_doc = [word.lower() if word.isupper() else word for word in words]
    return " ".join(processed_doc)

documents = [clean_doc(d) for d in documents]

# coherence score

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
# Tokenize your documents
tokenized_docs = [doc.split() for doc in documents]  # or use a tokenizer for more complex preprocessing

# Create a dictionary from the tokenized documents
dictionary = Dictionary(tokenized_docs)


# set random seed
import numpy as np
import random

my_seed=99
random.seed(my_seed)
np.random.seed(my_seed)

## soft kmeansimport numpy as np
import numpy as np


# sentence transformer

from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('TurkuNLP/sbert-cased-finnish-paraphrase')


vectorizer_model = CountVectorizer(stop_words=stop, min_df=5, max_df=0.95)
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

import hdbscan
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# different model possibilities:
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=30,        # Minimum cluster size
    min_samples=10,              # Minimum samples to be a core point
    metric='euclidean',            # Distance metric
    cluster_selection_method='eom',  # Cluster selection method
    prediction_data=True        # Enable prediction for unseen data
)


k_model = KMeans(n_clusters=ntopics, max_iter = 300, n_init=30, random_state=my_seed)
pca_model = PCA(n_components=5, random_state=my_seed)
umap_model = UMAP(n_neighbors=20, n_components=7, min_dist=0.0, metric='cosine', random_state=my_seed)


import h5py

# load in embeddings

if embedding_model == 'TurkuNLP/sbert-cased-finnish-paraphrase':
    epath="/scratch/project_2008526/telmap/suomi24/turkunlp_train_embeddings.h5"
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

# topic model 
topic_model = BERTopic(embedding_model=embedding_model,
                    ctfidf_model = ctfidf_model,
                    hdbscan_model =soft_kmeans_model,
                    umap_model =umap_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics=ntopics, # n topics
                    top_n_words=20,
                    min_topic_size=10,
                    calculate_probabilities=True,
                    n_gram_range=(1,3))


print("fitting embeddings", embedding_model, "with", dim_model_name,"and" , cluster_model_name ,"\n")
topics, probs = topic_model.fit_transform(documents, embeddings)
    
    
print(topics[:100])
#embedding_model = 'TurkuNLP/sbert-cased-finnish-paraphrase'
spath = fpath.split("corpus")[0] +"bertopicmodel/model_"+embedding_model.split("/")[-1]+"_" + dim_model_name+"_" + cluster_model_name

# make directory
import os
os.makedirs(spath, exist_ok=True)
print("\nsaving to", spath)
topic_model.save(spath+"/picklemodel", serialization="pickle", save_ctfidf=True)
#topic_model.save(spath, serialization="safetensors", save_ctfidf=True)


# predict new documents (not used)
# Predict topics for new documents
new_topics, new_probs = topic_model.transform(documents2)

#save predictions
res_df=pd.DataFrame()
res_df["document"] = documents2
res_df["topic"] = new_topics
df_path=spath+"predictions.csv"
res_df.to_csv(df_path, sep="\t", index=False)

# compute coherence score
tops = set(topics)
top_words_per_topic = [[word for word, _ in topic_model.get_topic(topic)] for topic in tops if topic != -1]

coherence_model = CoherenceModel(
    topics=top_words_per_topic,
    texts=[doc.split() for doc in documents],  # Tokenized documents
    dictionary=dictionary, 
    coherence='c_v'   # 'c_v' is popular for topic models, but you can experiment with others
)

# Calculate the coherence score
coherence_score = coherence_model.get_coherence()
print("Coherence Score:", coherence_score)
print("\n")

print("Done!\n")
