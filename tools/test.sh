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
  --name mnist \
  --global-train-ratios 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 |
#  --train-examples
#  --valid-examples
setup/nodes/google-fl.py \
  --nb-nodes 100 \
  --local-shards 2 \
  --shard-size 250 |
#setup/topology/d-cliques/ideal.py \
#  --interclique fully-connected |
setup/topology/ring.py \
  --metric dissimilarity |
setup/model/linear.py |
simulate/algorithm/d_sgd.py \
  --batch-size 128 |
simulate/logger.py \
  --nb-processes 2 |
simulate/run.py \
  --nb-epochs 1
