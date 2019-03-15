#!/bin/bash
#MSUB -l nodes=1:ppn=1,walltime=04:00:00
#MSUB -M hyunkim2015@u.northwestern.edu
#MSUB -m abe
#MSUB -N StagHunt

# Load modules
module load python/anaconda3

# Working directory
# cd $PBS_O_WORKDIR
cd ~/StagHunt

# Command
python 190315_hyperparameters_stag_hunt_n_dimensional.py