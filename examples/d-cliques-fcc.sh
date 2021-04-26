#!/usr/bin/env bash
TOOLS=../tools; CWD="$(pwd)"; cd $TOOLS
BSZS='
    128
    '
LRS='
    0.1
    '
for BSZ in $BSZS; 
    do for LR in $LRS;
        do python simulate.py --nb-nodes 100 --nb-epochs 100 --local-classes 1 --seed 1 --nodes-per-class 10 10 10 10 10 10 10 10 10 10 --global-train-ratios 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 0.802568 --dist-optimization d-psgd --topology fully-connected-cliques --metric dissimilarity --learning-momentum 0. --sync-per-mini-batch 1 --results-directory $CWD/all --learning-rate $LR --batch-size $BSZ "$@" --parallel-training --nb-workers 10 --dataset mnist --model linear --clique-gradient --initial-averaging
    done;
done;

