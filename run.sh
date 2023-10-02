#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "gpt2-imdb-2-emotions"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs

while getopts r: flag
do
    case "${flag}" in
        r) run_config=${OPTARG};;
    esac
done

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export WANDB_API_KEY=e279feeab3d602ab530e4eb23df8ac3ff3763461

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
wandb offline
mamba activate sentiment-tuning
#wandb login --relogin e279feeab3d602ab530e4eb23df8ac3ff3763461
python general_sentiment_tuning.py $run_config

