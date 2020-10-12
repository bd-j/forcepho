#!/usr/bin/env bash

# With Slurm heterogenous jobs, we can have the server running in one allocation
# and the client running in another, and still have them communicate in the
# same MPI_COMM_WORLD.
# This is useful because the server needs CPU and memory but not GPU,
# while the clients need GPU but not much CPU or memory.
# The risk is that some facilities don't have their GPU and CPU nodes
# set up for cross-talk via MPI!  One can use launch_parallel.sh
# in that case.

time="0:10"
gpus_per_node=2
server_resources="-N1 -n1 -c10 --mem=64G -p cca,gen"
client_resources="-N1 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu=2G --gres=gpu:${gpus_per_node} -p gpu"
server_script="python -m forcepho.dispatcher"
client_script="./client_shim.sh"

export FORCEPHO_GPUS_PER_NODE=$gpus_per_node
# This srun syntax with the ":" launches a "heterogeneous job"
sbatch -t $time $server_resources : $client_resources --wrap "srun ${server_script} : ${client_script}"
#sbatch -t 0:10 -N2 -n2 -c1 -p cca --wrap "mpirun -np 2 ./mpi_demo.py"
