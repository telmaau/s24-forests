#!/scratch/project_2008526/telmap/myvenv python

#%%
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import gensim
from gensim import corpora
import sys
import random
import pandas as pd
import os
import gzip
import glob
import re

from gensim.models import TfidfModel



fpath=sys.argv[1] # path to the csv file


print('Using lemmas from suomi24 data. Lowercased, numbers removed and punctuation cleaned.')
print(f'Data from {fpath}.\n')


# stopwords
custom_stop_words = ["eli","vain", "vaan","siksi","mä","sä","tällä", "silloin", "joku","jossain","se","siellä","ihan","vielä","mikään","of","and","if","the","på","och","by","myös","jo","make","van","jokin","in","sitten","voida"]

stop = stopwords.words("finnish")
stop.extend(custom_stop_words)



# functions

def clean(doc, stop,text_string=True):
    """
    doc: a text document
    stop: stopwords
    list: text as a string (if False, list)
    """
    # into list
    if text_string ==True:
        doc=doc.split(" ")

    # remove special characters
    doc_punc = [re.sub("[.,:;!?]", " ", d) for d in doc]
    doc_letter =[re.sub("[^a-zäöüåA-ZÅÄÖÜß]", "", d.strip()) for d in doc_punc]

    #lowercase
    doc_low = [d.lower() for d in doc_letter]

    # remove html addresses:
    html_free = [d for d in doc_low if "http" not in d and "www" not in d]

    #remove stopwords and drop 1-letter words
    stop_free = " ".join([i for i in html_free if i not in stop and len(i)>1])

    return stop_free


## data

texts_df = pd.read_csv(fpath, sep="\t")
texts_df = texts_df.drop_duplicates(subset="text")
texts_df['text_preprocessed'] = texts_df['lemmas'].apply(
    lambda x: clean(x, stop)
)



# bigrams
docs = [d.split() for d in texts_df['text_preprocessed'] ]
print("all docs:",len(docs))

bigram = gensim.models.Phrases(docs, min_count=25, threshold=150)


bigram_mod = gensim.models.phrases.Phraser(bigram)
def create_bigrams(texts):
    bg_list= [bigram_mod[doc] for doc in texts]
    bg_string = [" ".join(b) for b in bg_list]
    return bg_string

# add bigrams to df
data_bigrams = create_bigrams(docs)
texts_df["bigrams"] =data_bigrams
texts_df["length"] =  [len(d.split() ) for d in data_bigrams]


# filter out years before 2014 and clean metadata

def clean_metadata(meta):
    
    textlist = meta.split()
    
    id_dict={"date":"","id":"","comment_id":"","time":"", "parent_id":"","author":"", "msg_type":""}
    
    for i,t in enumerate(textlist):
        if "date=" in t:
            date= t.replace("date=","")
            id_dict["date"] = date
        elif t.startswith("time="):
            time=t.replace("time=","")
            id_dict["time"] =time
        elif t.startswith("id="):#
            c_id = t.replace("id=","")
            id_dict["id"] = c_id
        elif t.startswith("comment_id="):#
            com_id = t.replace("comment_id=","")
            id_dict["comment_id"] = com_id
        elif "author=" in t:
            
            author= t.replace("author=","")
            if "parent_comment_id=" not in textlist[i+1] :
                author = author + " "+textlist[i+1]
            id_dict["author"] =author
        elif "parent_comment_id=" in t:
            p_id = t.replace("parent_comment_id=" ,"")
            id_dict["parent_id"] = p_id
        elif "msg_type=" in t:
            type_id = t.replace("msg_type=" ,"")
            id_dict["msg_type"] = type_id
    return  id_dict

cleaned_meta = texts_df['info'].apply(clean_metadata)
print(len(cleaned_meta), len(texts_df))
print(cleaned_meta[0],"\n")


metadatadf = pd.json_normalize(cleaned_meta)
df2= pd.concat([texts_df.reset_index(),metadatadf], axis=1, join="inner")
df2["year"] =pd.DatetimeIndex(df2['date']).year
df2.sort_values(by="date",ascending=True, inplace=True)
#df2.to_csv(fpath.replace(".csv","len2.csv"), sep="\t",index=False)

# read in train data
trainpath= "/scratch/project_2008526/telmap/suomi24/corpus/s24_forest_train.csv"
df = pd.read_csv(trainpath, sep="\t")
documents=list(df["text_preprocessed"].drop_duplicates())
print("Training on", len(documents), "documents. \n")

# data to train on
#data_long=df2[(df2["length"]>5) & (df2["year"]>2013)]["bigrams"]
texts= [d.split() for d in documents]
#texts = [d.split() for d in data_long]



#print("nr of docs, len > 5 tokens:", len(texts))
print("\n")


print(f'---------------------------\nStarting to build the model')


# dictionary and corpus
gensim_dictionary = corpora.Dictionary(texts)
# Print the number of words in the dictionary
num_words = len(gensim_dictionary)
print(f'Number of words in the dictionary before filtering: {num_words}')
gensim_dictionary.filter_extremes(no_below=5, no_above=0.99, keep_n=200000)
print("no below 5, no above .99")


bow_corpus = [gensim_dictionary.doc2bow(text, allow_update=False) for text in texts]
id_words = [[(gensim_dictionary[id], count) for id, count in line] for line in bow_corpus]

# tf idf, alternative to bow
tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = tfidf_model[bow_corpus]


# Print the number of words in the dictionary
num_words = len(gensim_dictionary)
print(f'Number of words in the dictionary: {num_words}')


# save them if you want
from gensim.corpora import MmCorpus
#DICT_PATH="your-path-to/s24_topicmodel_gensim_bow.dict"
#CORPUS_PATH="your-path-to/s24_topicmodel_gensim_bow.mm"
#gensim_dictionary.save(DICT_PATH)
#MmCorpus.serialize(CORPUS_PATH, tfidf_corpus)



# use LDAmulticore  for faster processing
from pprint import pprint
from gensim.models import LdaMulticore
# initiate the model


## loop with different nr of topics:
from gensim.test.utils import datapath
path= "your-folder-path" # path to the folder where you want to save the models

from gensim.models.coherencemodel import CoherenceModel

# hyperparams
param_dict={'alpha':"asymmetric" , 'decay': 0.50001, 'offset': 64, 'eta': None, 'gamma_threshold': 0.0008, 'minimum_probability': 0.02}
print(param_dict)
print("passes 30, iterations 50 \n")

def build_lda_models(input_data, name, k, coherence=False):
    #np.random.seed(99)
    # Train the model
    # Train the LDA model using LdaMulticore
    lda = LdaMulticore(
        corpus=input_data,
        id2word=gensim_dictionary,
        num_topics=k,        # Set the number of topics
        passes=30,           # Number of passes through the corpus during training
        workers=4,           # Number of CPU cores to use
        random_state=99,     # Random seed for reproducibility
        per_word_topics=True,
        iterations=50,
        **param_dict
    )
    # Save the model

    tempfile = datapath( path + "/models/LDA_"
        + name + "_" + str(k)) +"bigram.model"
    
    print("saving", str(tempfile))
    lda.save(tempfile)
    pprint(lda.print_topics(num_topics=k, num_words=10))

    if coherence ==True:
        # print coherence
        coherence_lda_bow = CoherenceModel(model=lda,
            texts=texts,
            dictionary=gensim_dictionary,
            coherence='c_v')
        coherence_lda = coherence_lda_bow.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
    print("")

import numpy as np

numTopicsList = np.arange(25,251,25)

#coherenceList_bow = []
for k in numTopicsList:
    print(k)
    build_lda_models(bow_corpus, "bow_gensim", k, coherence =True)

print("")


