#!/usr/bin/env bash
# Path to current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Add current working directory to executable namespace
export PATH=$PATH:$SCRIPT_DIR/../
# Setup root directory for resolution of imports:
# the path of all local python libraries are relative to this
export PYTHONPATH=$SCRIPT_DIR/../

cd $SCRIPT_DIR/../

for C_GRAD in '--clique-gradient' ''; do
# Each command outputs the run directory, which is then used
# by the next command to add parameters and generate information
# used by the simulator. For a list of available options for each
# command, run 'export PYTHONPATH=.; <command> --help'.
setup/meta.py \
  --results-directory $SCRIPT_DIR/all \
  --log INFO |
setup/dataset.py \
  --name mnist \
  --train-examples-per-class 4500 4500 4500 4500 4500 4500 4500 4500 4500 4500 |
setup/nodes.py \
   --nb-nodes 20 \
   --nodes-per-class 2 2 2 2 2 2 2 2 2 2  \
   --local-classes 1 |
setup/topology/d_cliques/ideal.py \
  --remove-clique-edges 1 \
  --interclique fully-connected |
setup/model/linear.py |
simulate/algorithm/d_sgd.py \
  $C_GRAD \
  --batch-size 125 |
simulate/logger.py \
  --accuracy-logging-interval 1\
  --nb-processes 2 |
simulate/run.py \
  --nb-epochs 2;
done
