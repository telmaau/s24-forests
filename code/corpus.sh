#!/bin/bash

#SBATCH --job-name=seedLING_corpus        # Job name
#SBATCH --account=project_2008526        # Billing project, has to be defined!
#SBATCH --time=10:00:00             # Max. duration of the job
#SBATCH --mem-per-cpu=24G            # Memory to reserve per core
#SBATCH --partition=small          # Job queue (partition)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --mail-type=BEGIN          # Uncomment to enable mail

#module load python/3.9.7            # Load required modules
source /your-folder/.venv/bin/activate

## Directory containing the files
directory="/scratch/project_2008526/eltuom/suomi24_conllu_folders/alkuperaiset"

srun python3 -i compileCorpus_csv.py    # Run program using requested resources
    


