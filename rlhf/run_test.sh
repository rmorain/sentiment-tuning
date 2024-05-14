#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -J "run_test"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --qos=cs
#
wandb enabled
accelerate launch --config_file=multi_gpu.yaml --num_processes 8 rlhf/run_test.py 15
# python rlhf/run_test.py
