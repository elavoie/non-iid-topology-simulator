#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import math
from random import Random
from copy import deepcopy
import setup.meta as m
import setup.nodes
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights
import interclique

def cliques(_nodes, params):
    max_clique_size = params['topology']['max-clique-size']
    nodes = set([ n['rank'] for n in _nodes ])
    rand = Random()
    rand.seed(params['meta']['seed'])

    cliques = []
    while len(nodes) > max_clique_size:
        c = rand.sample(nodes, max_clique_size)
        nodes.difference_update(c)
        cliques.append(list(c))
    cliques.append(list(nodes))

    edges = {}
    for c in cliques:
        ranks = set(c)
        for rank in ranks:
            edges[rank] = ranks.difference([rank])

    return cliques, edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a D-Cliques Topology.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--interclique', type=str, default='fully-connected',
      choices=['ring', 'fractal', 'smallworld', 'fully-connected'],
      help="Interclique topology (default: 'fully-connected')")
    parser.add_argument('--weights', type=str, default='metropolis-hasting', 
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    parser.add_argument('--max-clique-size', type=int, default=30, metavar='N',
        help='Maximum number of nodes in a clique (default: 30)')
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    topology_params = {
        'name': 'd-cliques/random-cliques',
        'weights': args.weights,
        'interclique-topology': args.interclique,
        'max-clique-size': args.max_clique_size
    }
    m.extend(rundir, 'topology', topology_params)

    cliques, intra_edges  = cliques(nodes, m.params(rundir))
    # In contrast to pseudo code of paper, the edge set is directly
    # extended by the interclique (interconnect) method. 'edges' therefore
    # both includes the intraclique and interclique edges.
    logging.info(len(cliques))
    edges = interclique.get(args.interclique)(cliques, intra_edges, params)

    topology = {
      'edges': { rank: list(edges[rank]) for rank in edges },
      'weights': compute_weights(nodes, edges, topology_params),
      'cliques': cliques
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
