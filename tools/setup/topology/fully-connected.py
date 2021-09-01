#!/usr/bin/env python
import os
import argparse
import logging
import json
import setup.meta as m
from setup.topology.weights import compute_weights

def create(nodes):
    all_ranks = { n["rank"] for n in nodes }
    return { x: list(all_ranks.difference({x})) for x in all_ranks }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Fully-Connected Topology.')
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

    params = {
        'name': 'fully-connected',
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', params)

    edges = create(nodes)
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, params),
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
