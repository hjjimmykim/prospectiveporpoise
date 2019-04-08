#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=1day
#SBATCH --time=23:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=hyunjk@princeton.edu
#SBATCH --job-name=chicken.sh
#SBATCH --array=1-1

module load python/3.5.2

cd ~/prospectiveporpoise

python 190315_hyperparameters_chicken_n_dimensional.py
