#!/usr/bin/env bash
# Path to current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Add current working directory to executable namespace
export PATH=$PATH:./
# Setup root directory for resolution of imports:
# the path of all local python libraries are relative to this
export PYTHONPATH=$SCRIPT_DIR

cd $SCRIPT_DIR

# Each command outputs the run directory, which is then used
# by the next command to add parameters and generate information
# used by the simulator. For a list of available options for each
# command, run 'export PYTHONPATH=.; <command> --help'.
setup/meta.py \
  --results-directory all \
  --log INFO |
setup/dataset.py \
  --name cifar10 |
setup/nodes/google-fl.py \
  --nb-nodes 100 \
  --local-shards 2 \
  --shard-size 200 |
#setup/nodes.py \
#   --nb-nodes 100 \
#   --nodes-per-class 10 10 10 10 10 10 10 10 10 10 \
#   --local-classes 1 |
setup/topology/d_cliques/centralized-greedy.py \
  --interclique ring \
  --max-clique-size 20 |
#setup/topology/fully-connected.py |
#  --metric dissimilarity |
#setup/topology/ring.py \
#  --metric dissimilarity |
setup/model/gn_lenet.py |
simulate/algorithm/d_sgd.py \
  --clique-gradient \
  --batch-size 125 |
simulate/logger.py \
  --accuracy-logging-interval 10\
  --nb-processes 2 |
simulate/run.py \
  --nb-epochs 50
