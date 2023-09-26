#!/bin/bash

#SALLOC --time=10:00:00   # walltime
#SALLOC --ntasks=16   # number of processor cores (i.e. tasks)
#SALLOC --nodes=1   # number of nodes
#SALLOC --gpus=1
#SALLOC --mem-per-cpu=1024M   # memory per CPU core
#SALLOC -J "test-gst"   # job name
#SALLOC --mail-user=rmorain2@byu.edu   # email address
#SALLOC --qos=cs

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export WANDB_API_KEY=e279feeab3d602ab530e4eb23df8ac3ff3763461

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
wandb login --relogin $WANDB_API_KEY
