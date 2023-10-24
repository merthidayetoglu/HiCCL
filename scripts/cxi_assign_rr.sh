#!/usr/bin/env bash

num_cxi=8

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
fi

cxi_id=$((_MPI_RANKID % num_cxi))

export FI_CXI_DEVICE_NAME=cxi$cxi_id

"$@"
