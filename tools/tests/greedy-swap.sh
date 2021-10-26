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
setup/meta.py \
  --script $SCRIPT_DIR/`basename "$0"` \
  --results-directory $SCRIPT_DIR/ \
  --log INFO |
setup/dataset.py \
  --validation-examples-per-class 100 100 100 100 100 100 100 100 100 100 \
  --train-examples-per-class 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 \
  --name mnist |
setup/nodes/google-fl.py \
  --nb-nodes 100 \
  --local-shards 2 \
  --shard-size 200 |
setup/topology/d_cliques/greedy_swap.py \
  --metric euclidean \
  --max-clique-size 10|
setup/model/linear.py |
simulate/algorithm/d_sgd.py \
  --batch-size 128 |
simulate/logger.py \
  --accuracy-logging-interval 10\
  --nb-processes 8 |
simulate/run.py \
  --nb-epochs 100