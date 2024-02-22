#!/bin/bash

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbkf-delta-gpu
#SBATCH --time=00:30:00
### GPU options ###
#SBATCH --gpus-per-node=4

date

scontrol show job ${SLURM_JOBID}

module -t list

pattern=3
count=$((2 ** 25))
numstripe=4
ringnodes=4
pipedepth=128
warmup=5
numiter=10

srun ./HiCCL $pattern $count $numstripe $ringnodes $pipedepth $warmup $numiter

date
