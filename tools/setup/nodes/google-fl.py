#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import json
import time
import argparse
import logging
import socket
from random import Random
import setup.dataset as ds
import setup.meta as m
import math

def params(_params):
    if type(_params) == str:
        params = m.params(_params)
    elif type(_params) == dict:
        assert 'nodes' in _params.keys(), "Invalid _params dictionary, should have " +\
        "a 'nodes' property"
        params = _params['nodes']
    else:
        raise Exception('Invalid _params parameter, should be path to a run directory or a ' +\
              'dictionary instead of {}'.format(_params))
    return validate(params)

def validate(nodes_params):
    # TODO
    return nodes_params

# We avoid explicitly passing and saving individual indexes of training examples
# by using ranges over the randomized list of indexes, that is the same for all nodes.
# 
# Generate the ranges centrally and quickly, actual indexes of examples are selected
# with dataset.partition.
def assign_ranges (params, expected_nb_shards):
    meta_params = params['meta']
    node_params = params['nodes']
    dataset_params = params['dataset']
    seed = meta_params['seed']
    nb_nodes = node_params['nb-nodes']
    shard_nb = node_params['local-shards']
    rand = Random()
    rand.seed(seed)

    # Create shards
    examples_per_class = dataset_params['train-examples-per-class'] 
    shard_size = node_params['shard-size']
    shards = []
    remaining = [ s for s in examples_per_class ]
    c = 0
    for _ in range(int(expected_nb_shards)):
        assert sum(remaining) >= shard_size, "Insufficient number of available examples"
        shard = {}
        assigned = 0
        while assigned < shard_size:
            if remaining[c] == 0:
                c += 1
            s = min(shard_size - assigned, remaining[c])
            remaining[c] -= s
            shard[c] = s 
            assigned += s
        shards.append(shard)
    assert sum(remaining) == 0, "Remaining unassigned examples"

    # Assign shards to nodes
    rand.shuffle(shards)
    nodes = [ { "rank": i, "classes": None, "samples": None } for i in range(nb_nodes) ]

    # save [start, end[ for each class of every node where:
    # 'start' is the inclusive start index
    # 'end' is the exclusive end index
    start = [ 0 for i in range(10) ]
    for n in nodes:
        n['classes'] = [ 0 for _ in range(dataset_params['nb-classes']) ]
        local_shards = shards[n['rank']*shard_nb:(n['rank']*shard_nb)+shard_nb]
        end = [ x for x in start ]
        for shard in local_shards:
            for k in shard.keys():
                n['classes'][k] = 1.0
                end[k] += shard[k] 
        n['samples'] = [(start[c], end[c]) for c in range(10)]
        start = end

    assert all([ t1 == t2 for t1,t2 in zip(end, node_params['total-of-examples']) ]),\
        "Expected {} total samples while we assigned {}".format(node_params['total-of-examples'], end)

    multiples = [ 0 for _ in range(10) ] 
    for n in nodes:
        for c in range(10):
            multiples[c] += n["classes"][c]
    logging.info('assign_classes: classes represented times {}'.format(multiples))

    return nodes

def size(node):
    assert 'samples' in node.keys(), "Expected 'samples' key on node"
    samples = node['samples']
    total = 0
    for start, end in samples:
        total += end-start
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partition the Dataset between Nodes following '+\
            ' the non-iid scheme in http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf ' +\
            " (Google Federated Learning's founding paper).")
    parser.add_argument('--rundir', type=str, default=None,
        help='Directory of the run in which to save the partition options.')
    parser.add_argument('--nb-nodes', type=int, default=1, metavar='N',
        help='number of nodes (default: 1)')
    parser.add_argument('--shard-size', type=int, default=250, metavar='N',
        help='Size of shards in which to divide the examples used for each class. (default: 250)')
    parser.add_argument('--local-shards', type=int, default=2,
        help='Number of shards per node. (default: 2)')
    parser.add_argument('--name', type=str, default='',
        help='Name given to this partition scheme.')

    args = parser.parse_args()
    rundir = m.rundir(args)

    params = m.params(rundir)
    meta_params = m.params(rundir, 'meta')
    dataset_params = ds.validate(params['dataset'])

    logging.basicConfig(level=getattr(logging, meta_params['log'].upper(), None))

    examples_per_class = [ t for t in dataset_params['train-examples-per-class'] ]
    # Make sure the dataset splits evenly in shards of size 'shard-size'
    assert sum(examples_per_class) % args.shard_size == 0,\
        'Total distinct train examples ({}) do not split evenly in shards of size {}'.format(\
        sum(examples_per_class), args.shard_size)

    # Make sure the resulting number of shards splits evenly between nodes
    nb_shards = int(sum(examples_per_class) / args.shard_size)
    assert nb_shards % args.nb_nodes == 0,\
        'Total number of shards ({}) does not split evenly in {} nodes'.format(\
        nb_shards, args.nb_nodes)

    # Make sure the dataset splits in the total number of shards requested
    assert sum(examples_per_class) == args.nb_nodes * args.local_shards * args.shard_size,\
        'Invalid combination of nb-nodes, local-shards, and shard-size, ' +\
        'nb-nodes * local-shards * shard-size should be equal to sum(train-examples-per-class)'

    node_params = {
        'name': args.name,
        'nb-nodes': args.nb_nodes,
        'local-shards': args.local_shards,
        'shard-size': args.shard_size,
        'total-of-examples': [ e for e in examples_per_class ]
    }
    m.extend(rundir, 'nodes', node_params)

    nodes = assign_ranges(m.params(rundir), nb_shards)
    with open(os.path.join(rundir, 'nodes.json'), 'w+') as node_file:
        json.dump(nodes, node_file)

    if args.rundir is None:
        print(rundir)
