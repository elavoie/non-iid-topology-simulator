#!/usr/bin/env bash
# Path to current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Setup root directory for resolution of imports:
# the path of all local python libraries are relative to this
export PYTHONPATH=$SCRIPT_DIR

python $@
