#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import math
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights

def create(nodes, params):
    edges = { n['rank']: set() for n in nodes }
    max_offset = math.floor(math.log(len(nodes)-1) / math.log(2))
    for n in nodes:
        for offset in range(max_offset+1):
            i = n['rank']
            j = (i + 2**offset) % len(nodes)
            edges[i].add(j)
            edges[j].add(i)

    #for rank in edges:
    #    assert len(edges[rank]) == (max_offset+1), "Invalid number of edges, node {} has {} edges instead of the {} expected".format(rank, len(edges[rank]), max_offset+1)

    return { rank: list(edges[rank]) for rank in edges }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an Expander Graph Topology, following https://arxiv.org/abs/2110.13363.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--weights', type=str, default='metropolis-hasting', 
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    topology_params = {
        'name': 'expander-graph',
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir)

    edges = create(nodes, params)
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, topology_params),
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
