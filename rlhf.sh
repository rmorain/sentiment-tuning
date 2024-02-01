#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "IMDB sentiment tuning"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#
wandb enabled
accelerate launch --config_file=multi_gpu.yaml --num_processes 2 rlhf.py 2 
