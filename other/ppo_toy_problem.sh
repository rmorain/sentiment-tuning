#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=256G   # memory per CPU core
#SBATCH -J "debug: ppo toy_problem"   # job name
#SBATCH --mail-user=rmorain2@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#

python ppo_toy_problem.py
