#!/bin/bash

#SBATCH --time=23:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -J "Toxicity"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#SBATCH --output=/home/rmorain2/sentiment_tuning/slurm-logs/slurm-%j.out
#
wandb enabled
accelerate launch --config_file=multi_gpu.yaml --num_processes 8 rlhf/toxicity.py 2
# python rlhf/rlhf.py 8
