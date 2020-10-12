#!/usr/bin/env bash

# Assign one GPU to this client
# Selects from the GPUs on this node in a round-robin fashion
export CUDA_VISIBLE_DEVICES=$(( ${SLURM_LOCALID} % ${FORCEPHO_GPUS_PER_NODE} ))
echo "Client: host $(hostname), rank ${SLURM_PROCID}, gpu ${CUDA_VISIBLE_DEVICES}"
python -m forcepho.dispatcher

# A GPU may be assigned to more than one client, but this probably only helps with CUDA MPS,
# and indeed may crash if the GPUs are in exclusive mode.
# The client could select its own GPU instead of relying on this shim layer,
# but one might worry that top-level imports would see the wrong GPU before it could be changed.
