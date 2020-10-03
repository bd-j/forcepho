#!/usr/bin/env bash

# This script launches the server on one node and the clients
# on the rest.  The GPUs on the server node are wasted.

# An enhancement would be to use manual host specification to
# launch the server on a client node.

time="0:10"
gpus_per_node=2
resources="-N1 -n2 --cpus-per-task=3 --mem-per-cpu=3G -m cyclic --gres=gpu:${gpus_per_node} -p gpu"
script="./client_shim.sh"

export FORCEPHO_GPUS_PER_NODE=$gpus_per_node
sbatch -t $time $resources --wrap "srun ${script}"
