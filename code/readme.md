# Code instructions
You can replicate the study by running
1) corpus.sh to compile the corpus from raw s24 data files
2) run_k_search.sh to estimate the k number of topics used in the kmeans models.
3) run_bert_topics.sh to train BERTopic models.


- data_prep.ipynb shows how the code was split into train and test set
- bertopic_finnish_umap_kmeans_analysis_clean.ipynb contains analysis of the BERTopic model with umap, kmeans and turkunlp sentence embeddings. You can use the same notebook with other models by changing the model name.

N.B. Make sure that all the data and folder paths are correct according to your folder structure!



## Trials with gensim
I included also the gensim_topics.py and tm_evaluation, although they did not seem to yield very good topics. Thus, not all the gensim models were saved, but if you want to experiment further, you can use the gensim_topics.py as a starting point.
