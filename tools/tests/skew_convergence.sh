#!/usr/bin/env bash
# Path to current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TOOLS=$SCRIPT_DIR/..; cd $TOOLS

# Add current working directory to executable namespace
export PATH=$PATH:$TOOLS
# Setup root directory for resolution of imports:
# the path of all local python libraries are relative to this
export PYTHONPATH=$TOOLS

# Each command outputs the run directory, which is then used
# by the next command to add parameters and generate information
# used by the simulator. For a list of available options for each
# command, run 'export PYTHONPATH=.; <command> --help'.
for SEED in 1 2 3; do
cat $(setup/meta.py \
  --script $SCRIPT_DIR/`basename "$0"` \
  --results-directory $SCRIPT_DIR/skews \
  --seed $SEED |
setup/dataset.py \
  --name mnist \
  --train-examples-per-class 4900 5600 4900 5100 4800 4500 4900 5200 4800 4900 | 
setup/nodes/google-fl.py \
  --name 2-shards-uneq-classes \
  --nb-nodes 100 \
  --local-shards 2 \
  --shard-size 248 |
setup/topology/d_cliques/greedy_swap.py \
  --interclique fully-connected \
  --max-clique-size 10 \
  --max-steps 1000)/events/global.jsonlines;
done
