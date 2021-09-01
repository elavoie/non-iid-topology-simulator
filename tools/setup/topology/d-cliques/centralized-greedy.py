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

def skew(d1, d2):
    def valid(d):
        return all(map(lambda x: x >= 0 and x <= 1, d1)) and\
               sum(d) >= 0.999999 and\
               sum(d) <= 1.000001

    assert len(d1) == len(d2), "Inconsistent length between {} and {}".format(d1,d2)
    assert valid(d1), "Invalid d1 ({}), all values should be between 0 and 1, and sum to 1"
    assert valid(d2), "Invalid d2 ({}), all values should be between 0 and 1, and sum to 1"

    return sum([ abs(x-y) for x,y in zip(d1,d2) ])

def dist(nodes):
    nbs = [ 0 for _ in nodes[0]['samples'] ]
    for n in nodes:
        samples = n['samples']
        for i in range(len(nbs)):
            r = samples[i]
            nbs[i] += (r[1]-r[0])
    total = sum(nbs)
    return [ float(x)/total for x in nbs ]


def cliques(_nodes, params):
    max_clique_size = params['topology']['max-clique-size']
    nodes = [ deepcopy(n) for n in _nodes ] 
    for n in nodes:
        n['dist'] = dist([n])

    cliques = []
    global_dist = dist(nodes)
    for n in nodes:
        best = math.inf 
        best_c = None
        for c in cliques:
            if len(c) >= max_clique_size:
                continue
            dist_c = dist(c)
            dist_new_c = dist(c + [n])
            new = skew(dist_new_c, global_dist)
            current = skew(dist_c, global_dist)
            if new < current and new < best:
                best = new
                best_c = c

        if best_c is not None:
            best_c.append(n)
        else:
            cliques.append([n])

    edges = {}
    clique_ranks = []
    for c in cliques:
        ranks = set([ n['rank'] for n in c ])
        clique_ranks.append(list(ranks))
        for rank in ranks:
            edges[rank] = ranks.difference([rank])

    return clique_ranks, edges

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
        'name': 'd-cliques/centralized-greedy',
        'weights': args.weights,
        'interclique-topology': args.interclique,
        'max-clique-size': args.max_clique_size
    }
    m.extend(rundir, 'topology', topology_params)

    cliques, intra_edges  = cliques(nodes, m.params(rundir))
    # In contrast to pseudo code of paper, the edge set is directly
    # extended by the interclique (interconnect) method. 'edges' therefore
    # both includes the intraclique and interclique edges.
    edges = interclique.get(args.interclique)(cliques, intra_edges, params)

    topology = {
      'edges': { rank: list(edges[rank]) for rank in edges },
      'weights': compute_weights(nodes, edges, topology_params),
      'cliques': cliques
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
