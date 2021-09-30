#!/usr/bin/env bash
# Path to current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Add current working directory to executable namespace
export PATH=$PATH:$SCRIPT_DIR/../
# Setup root directory for resolution of imports:
# the path of all local python libraries are relative to this
export PYTHONPATH=$SCRIPT_DIR/../

cd $SCRIPT_DIR/../

# Each command outputs the run directory, which is then used
# by the next command to add parameters and generate information
# used by the simulator. For a list of available options for each
# command, run 'export PYTHONPATH=.; <command> --help'.
for MS in 10; do
  setup/meta.py \
    --results-directory $SCRIPT_DIR/all \
    --seed 1|
  setup/dataset.py \
    --name mnist |
  setup/nodes/google-fl.py \
    --nb-nodes 20 \
    --local-shards 10 \
    --shard-size 250 |
  setup/topology/d_cliques/bipartite.py \
    --interclique fully-connected \
    --max-clique-size $MS |
  setup/model/linear.py |
  simulate/algorithm/d_sgd.py \
    --batch-size 125 \
    --learning-momentum 0.0 \
    --learning-rate 0.1 \
    --clique-gradient |
  simulate/logger.py \
    --accuracy-logging-interval 2\
    --skip-testing\
    --nb-processes 2 |
  simulate/run.py \
    --nb-epochs 2;
done
