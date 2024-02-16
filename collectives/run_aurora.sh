#!/bin/bash

#PBS -l select=2:system=sunspot,place=scatter
#PBS -A CSC249ADCD01_CNDA
#PBS -l walltime=01:00:00
#PBS -N 2nodes_gpu
#PBS -k doe

export TZ='/usr/share/zoneinfo/US/Central'
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
#unset OMP_PLACES

date

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=12            # Number of MPI ranks per node
NDEPTH=8            # Number of hardware threads per rank, spacing between MPI ranks on a node
# NTHREADS=$OMP_NUM_THREADS # Number of OMP threads per rank, given to OMP_NUM_THREADS

export MPICH_GPU_SUPPORT_ENABLED=1

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_NODES=${NNODES}  TOTAL_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}  THREADS_PER_RANK=${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"

export MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE=1
PROC_LIST='list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99'

mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} --cpu-bind=$PROC_LIST gpu_tile_compact.sh ./cxi_assign_rr.sh ./HiCCL

date
