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
import metrics
from setup.topology.weights import compute_weights
import interclique
import utils

def cliques(nodes, params):
    # Random assignment
    max_clique_size = params['topology']['max-clique-size']
    ranks = set([ n['rank'] for n in nodes ])
    rand = Random()
    rand.seed(params['meta']['seed'])

    cliques = []
    while len(ranks) > max_clique_size:
        c = rand.sample(ranks, max_clique_size)
        ranks.difference_update(c)
        cliques.append(set(c))
    cliques.append(set(ranks))

    global_dist = metrics.dist(nodes)

    rand = Random()
    rand.seed(params['meta']['seed'])
    for k in range(params['topology']['max-steps']):
        c1,c2 = rand.sample(cliques, 2)
        c1_skew = metrics.skew(metrics.dist([ nodes[r] for r in c1 ]), global_dist)
        c2_skew = metrics.skew(metrics.dist([ nodes[r] for r in c2 ]), global_dist)
        baseline = c1_skew + c2_skew

        def updated(n1,c1,n2,c2):
            c1_updated = c1.difference([n1]).union([n2])
            c2_updated = c2.difference([n2]).union([n1])
            c1_updated_skew = metrics.skew(metrics.dist([ nodes[r] for r in c1_updated ]), global_dist)
            c2_updated_skew = metrics.skew(metrics.dist([ nodes[r] for r in c2_updated ]), global_dist)
            return c1_updated_skew + c2_updated_skew

        pairs = [ (n1,n2,updated(n1,c1,n2,c2)-baseline) for n1 in c1 for n2 in c2 ]
        improving = [ p for p in pairs if p[2] < 0 ]
        if len(improving) > 0:
            p = rand.sample(improving, 1)[0]
            c1.remove(p[0])
            c1.add(p[1])
            c2.remove(p[1])
            c2.add(p[0])

            # stats
            skews = [ metrics.skew(metrics.dist([ nodes[r] for r in c]), global_dist) for c in cliques ]
            last = k

    logging.info('last step {:3d} skew min {:.3f} max {:.3f} avg {:.3f}'.format(last, min(skews), max(skews), sum(skews)/len(skews)))

    edges = {}
    for c in cliques:
        ranks = set(c)
        for rank in ranks:
            edges[rank] = ranks.difference([rank])

    return [ list(c) for c in cliques ], edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a D-Cliques Topology with initial Random Cliques then Greedy Swaps.')
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
    parser.add_argument('--max-steps', type=int, default=1000, metavar='N',
        help='Maximum number of steps to consider after initial random clique generation (default: 1000).')
    parser.add_argument('--remove-clique-edges', type=int, default=0, metavar='N', 
            help="Remove X random edges from each clique. ( default: 0)")
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    topology_params = {
        'name': 'd-cliques/greedy-swap',
        'weights': args.weights,
        'interclique-topology': args.interclique,
        'max-clique-size': args.max_clique_size,
        'max-steps': args.max_steps,
        'remove-clique-edges': args.remove_clique_edges
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir) # reload extended

    cliques, intra_edges  = cliques(nodes, params)
    edges = interclique.get(args.interclique)(cliques, intra_edges, params)
    if topology_params['remove-clique-edges'] > 0: 
        edges, cliques = utils.remove_clique_edges(edges, cliques, params)

    topology = {
      'edges': { rank: list(edges[rank]) for rank in edges },
      'weights': compute_weights(nodes, edges, topology_params),
      'cliques': cliques
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
