#!/bin/bash
#SBATCH --job-name=bertopic_hdbscan
#SBATCH --account=ADD_ACCOUNT
#SBATCH --time=50:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20G            # Memory to reserve per core
##SBATCH -o %j.out
##SBATCH -e %j.err

file_path="add-your-file-path"

source /your-venv/bin/activate





## turku-nlp umap kmeans
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="umap"
cluster="kmeans"
srun python3 run_k_search.py $file_path $emb $dim $cluster

## turku-nlp pca kmeans
emb="TurkuNLP/sbert-cased-finnish-paraphrase"
dim="pca"
cluster="kmeans"
srun python3 run_k_search.py $file_path $emb $dim $cluster



## xlm-r umap kmeans
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="umap"
cluster="kmeans"
srun python3 run_k_search.py $file_path $emb $dim $cluster

## xlm-r pca kmeans
emb="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
dim="pca"
cluster="kmeans"
srun python3 run_k_search.py $file_path $emb $dim $cluster



##seff $SLURM_JOBID
