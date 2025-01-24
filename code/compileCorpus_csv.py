#!/users/telmpeur/suomi24/.venv python3


#import sys
import re
#import subprocess
import os
import glob
import gzip
#from collections import Counter
import pandas as pd



# functions
def clean_id(idline):
    """
    This function cleans the metadata line from the conll file
    that usually begins like that:
    ###C: <text comment_id=59203728 date=2013-01-03 datetime=2013-01-03 13:02:01 author=päästä parent_comment_id=0
    to collect the necessary info from it
    (my original idea was to save these as a dictionary, but due to some bugs it doesn't really work..)
    """
    cleaned_idline= idline.replace('"', '').replace("'", '')
    textlist = cleaned_idline.split()
    
    id_dict={"date":"","id":"","comment_id":"","time":"", "parent_id":"","author":"","text":""}
    
    for t in textlist:
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
            id_dict["author"] =author
        elif "parent_comment_id=" in t:
            p_id = t.replace("parent_comment_id=" ,"")
            id_dict["parent_id"] = p_id
    return cleaned_idline, id_dict
    

# save wanted posts as a csv file
def save_csv(meta,conll, filename):

    spath=filename.replace(".conll.gz","v1.csv")
    # append to file
    with gzip.open(filename, 'at', encoding='utf-8') as f:
        # first write the id line that contains information about the post
        #f.write(meta + "\n")
        text = []
        lemmas= []
        for i,sentence in enumerate(conll): # conll is a list of lines

            if len(sentence)==1: 
                # find text lines
                if "# text" in sentence[0]:
                    temp_text=sentence[0].replace("# text =","").strip()
                    text.append(temp_text)
            elif len(sentence) == 10: # conll line containing a token and related information
                temp_lemmas =sentence[2] # lemma 3rd column
                lemmas.append(temp_lemmas)
        temp=pd.DataFrame([{"text": " ".join(text),"info":meta, "lemmas":" ".join(lemmas)}])
        temp.to_csv(spath, mode='a', encoding='utf-8', header=False, index=False, sep="\t") # append to csv
                
            

def parse_gzip_conll(dat):
    with gzip.open(dat, 'rt') as f:
        feats = []  # the conll file content (features)
        id = None  # metadata about the text
        #meta = []
        for line in f:
            line = line.strip()
            
            # Start of a new document, include metadata
            if line.startswith("###C: <text") or line.startswith("###C: doc_id"):
                if id is not None:
                    # Yield the previous document's data
                    yield (id, feats)
                    feats = []  # Reset the features for the new document
                id = line  # Set the new document's metadata

            # End of a sentence, append an empty line
            elif "</sentence>" in line:
                feats.append(["</sentence>"])

            # add original text strings
            elif "# text" in line and id is not None:
                #feats.append(["\n"])
                feats.append([line])

        
            # Actual conll lines
            else:
                if line and not line.startswith("#"):  # Skip comments/metadata lines
                    feats.append(line.split("\t"))
                    
        # Yield the last document's data if any
        if id is not None:
            yield (id, feats)


# a function to check if any tokens in the post match our search words
def wordSearch(file, topicwords, fname):
    counter = 0
    fpath = "../data" # path to the folder where to save the data

    if not os.path.exists(fpath):
        os.mkdir(fpath)
        print("Created directory", fpath)
    
    
    # loop
    for meta, feats in parse_gzip_conll(file):

        # Check for matches in the sentence lemmas
        lemmas = [f[2] for f in feats if len(f) > 1]

        matches= list(set(lemmas) & set(topicwords))

        if len(matches)>0:
            counter += 1
            #print("match")
            if counter % 1000 == 0:
                print(f"Reached {counter} matches")

            new_id, metadict = clean_id(meta)

            # Write matching data to disk
            save_csv(new_id, feats, fname)

        # For testing the script, limit processing to the first 1000 matches
        #if counter > 1000:
            #print("Limit reached")
            #break

    print(f"Search complete. Matching texts were found {counter} times. \n")

# loop

print("Starting the collection: \n ")

#input datalist with all the original conll.gz files (one per year)
datalist = glob.glob("/scratch/project_2008526/eltuom/suomi24_conllu_folders/alkuperaiset/*")

#save with this filename
fname="/scratch/project_2008526/telmap/suomi24/corpus/s24_metsa_actors_new.conll.gz"

## create csv where to append metadata
id_dict={"date":"","id":"","comment_id":"","time":"", "parent_id":"","author":"","text":"","info":""}
metadf=pd.DataFrame([id_dict])
dfpath= fname.replace(".conll.gz","_new.csv")
metadf.to_csv(dfpath, index=False, sep="\t")


# create csv
spath=fname.replace(".conll.gz","_text2.csv")
df = pd.DataFrame(columns=["text","info","lemmas"])

df.to_csv(spath, mode='w', index=False, encoding='utf-8', sep="\t")

# topic words
tpath="/scratch/project_2008526/telmap/suomi24/metsa_hakusanat.csv"
topic_df=pd.read_csv(tpath, sep="\t")
topicwords= list(topic_df["sana"])

# print topic words
print("Using words:")
for tword in topicwords:
    print(tword)
print("\n")

# run
for d in datalist:
    year=d.split("s24_")[1]
    yearpattern= r'([0-9]{4})'
    ymatch=re.search(yearpattern,year)
    y=ymatch.group(0)
    
    if int(y) > 2010: # I didn't take years before 2010
        print(y+":",d)
        wordSearch(d, topicwords, fname)
