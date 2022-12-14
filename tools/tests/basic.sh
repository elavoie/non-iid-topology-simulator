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
  --results-directory $SCRIPT_DIR/all \
  --log INFO |
setup/dataset.py \
  --validation-examples-per-class 100 100 100 100 100 100 100 100 100 100 \
  --train-examples-per-class 5000 5000 5000 5000 5000 5000 5000 5000 5000 5000 \
  --name mnist |
setup/nodes/google-fl.py \
  --nb-nodes 2 \
  --local-shards 10 \
  --shard-size 2500 |
setup/topology/ring.py \
  --metric random |
setup/model/linear.py |
simulate/algorithm/d_sgd.py \
  --batch-size 125 |
simulate/logger.py \
  --accuracy-logging-interval 1\
  --nb-processes 2 |
simulate/run.py \
  --nb-epochs 1
