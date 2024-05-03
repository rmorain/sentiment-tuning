#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -J "run_test"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#
wandb enabled
python rlhf/run_test.py
