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
def assign_ranges (meta_params, node_params, dataset_params):
    seed = meta_params['seed']
    nb_nodes = node_params['nb-nodes']
    nodes_per_class = node_params['nodes-per-class']
    local_classes = node_params['local-classes']
    rand = Random()
    rand.seed(seed)
    remaining_classes = [ n for n in nodes_per_class ]

    # Sample without putting the previous samples back 
    # in the bag until empty, to guarantee coverage 
    # of all classes with few nodes or rare classes
    def sampler_gen (remaining_classes, k):
        while sum(remaining_classes) > 0:
            choices = []
            for c in range(10):
                if remaining_classes[c] > 0:
                    choices.append(c)
            s = rand.sample(choices, k)
            for c in s:
                remaining_classes[c] -= 1
            yield s

    def classes():
        samples = next(sampler)
        classes = [ 0. for c in range(10) ]
        for i in range(local_classes):
            classes[samples[i]] = 1.
        return classes

    sampler = sampler_gen(remaining_classes, local_classes)
    nodes = [ { "rank": i, "classes": classes() } for i in range(nb_nodes) ]
    multiples = [ 0 for _ in range(10) ] 
    for n in nodes:
        for c in range(10):
            multiples[c] += n["classes"][c]

    logging.info('assign_classes: classes represented times {}'.format(multiples))

    # Ensure each node has a disjoint set of examples
    # with all examples are assigned to nodes
    # Use the same (max) number for all classes to ensure
    # equal number of training examples regardless
    # of the class(es) chosen. Repeat examples when necessary.
    train_examples_not_used_in_val_set = [ ds.numbers[dataset_params['name']]['val_set_number_of_examples'][c] - 
                          math.floor(ds.numbers[dataset_params['name']]['val_set_number_of_examples'][c] * 
                                     dataset_params['validation-set-ratio']) for c in range(10) ]

    examples_per_class = max([ ds.numbers[dataset_params['name']]['train_set_number_of_examples'][c] + 
                               train_examples_not_used_in_val_set[c] for c in range(10) ])

    # save [start, end[ for each class of every node where:
    # 'start' is the inclusive start index
    # 'end' is the exclusive end index
    start = [ 0 for i in range(10) ]
    for n in nodes:
        end = [ start[c] + int(math.ceil(n["classes"][c] * node_params['local-train-ratios'][c] * examples_per_class))  
                for c in range(10) ]
        n['samples'] = [(start[c], end[c]) for c in range(10)]
        start = end

    return nodes, end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partition the Dataset between Nodes.')
    parser.add_argument('--rundir', type=str, default=None,
        help='Directory of the run in which to save the partition options.')
    parser.add_argument('--nb-nodes', type=int, default=1, metavar='N',
        help='number of nodes (default: 1)')
    parser.add_argument('--nodes-per-class', type=int, default=None, nargs='+',
        help='Number of nodes having examples of each class.')
    parser.add_argument('--local-classes', type=int, default=10,
        help='Number of classes represented in a single node (classes are ' + 
        'chosen randomly). (default: 10)')

    args = parser.parse_args()
    rundir = m.rundir(args)

    meta_params = m.params(rundir, 'meta')
    dataset_params = ds.validate(m.params(rundir, 'dataset'))

    logging.basicConfig(level=getattr(logging, meta_params['log'].upper(), None))

    len_classes = len(ds.numbers[dataset_params['name']]['classes'])
    assert args.local_classes >= 1 and args.local_classes <= 10,\
        "local-classes: should be between 1 and 10"

    if args.nodes_per_class is None:
        # By default, nodes per class should be balanced
        args.nodes_per_class = [ math.ceil( (args.nb_nodes / len_classes) * args.local_classes ) for i in range(10) ] 
    elif args.nodes_per_class is not None and len(args.nodes_per_class) != len_classes:
        print('Invalid number of nodes per class, expected {} integer values'.format(len_classes))
        sys.exit(1)
    elif sum(args.nodes_per_class) != args.nb_nodes * args.local_classes:
        print('Invalid number of nodes per class, should sum to nb-nodes * local-classes')
        sys.exit(1)

    node_params = {
        'nb-nodes': args.nb_nodes,
        'nodes-per-class': args.nodes_per_class,
        'local-classes': args.local_classes,
        'local-train-ratios': [ (1./n)*r for r,n in zip(dataset_params['global-train-ratios'], args.nodes_per_class) ],
        'total-of-examples': [ 0 for _ in dataset_params['global-train-ratios'] ]
    }

    nodes, total_of_examples = assign_ranges(meta_params, node_params, dataset_params)
    node_params['total-of-examples'] = total_of_examples

    m.extend(rundir, 'nodes', node_params)
    with open(os.path.join(rundir, 'nodes.json'), 'w+') as node_file:
        json.dump(nodes, node_file)

    if args.rundir is None:
        print(rundir)
