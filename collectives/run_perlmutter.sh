#!/bin/bash
#SBATCH -A m4301
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:01:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --array=0-28

date

module -t list

#export MPICH_OFI_NIC_VERBOSE=2
#export MPICH_ENV_DISPLAY=1

export SLURM_CPU_BIND="cores"

warmup=5
numiter=10

ringnodes=1
numstripe=1
stripeoffset=1
pipeoffset=1

for pattern in 8
do
for pipedepth in 128
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
#for size in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
for size in $SLURM_ARRAY_TASK_ID
do
  count=$((2**size))
  srun -N 4 --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none ./ExaComm $pattern $ringnodes $numstripe $stripeoffset $pipedepth $pipeoffset $count $warmup $numiter
done
done
done

date
