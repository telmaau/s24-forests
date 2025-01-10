#!/bin/bash
#SBATCH --job-name=bertopic_models
#SBATCH --account=ADD_ACCOUNT
#SBATCH --time=50:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20G            # Memory to reserve per core
##SBATCH -o %j.out
##SBATCH -e %j.err

file_path="/your-folder/path-to-file.csv"

source /your-python-venv/bin/activate

## umap = 175
## turku-pca 200
## xlm-pca 150

## turku-nlp umap hdbscan
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="umap"
cluster="hdbscan"
ntopics="175"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## turku-nlp umap kmeans
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="umap"
cluster="kmeans"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## turku-nlp pca kmeans
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="pca"
cluster="kmeans"
ntopics="200"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## turku-nlp pca hdbscan
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="pca"
cluster="hdbscan"
ntopics="200"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics


## xlm-r umap hdbscan
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="umap"
cluster="hdbscan"
ntopics="175"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## xlm-r umap kmeans
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="umap"
cluster="kmeans"
ntopics="175"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## xlm-r pca kmeans
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="pca"
cluster="kmeans"
ntopics="150"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics

## xlm-r pca hdbscan
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="pca"
cluster="hdbscan"
ntopics="150"
srun python3 train_models.py $file_path $emb $dim $cluster $ntopics



##seff $SLURM_JOBID
