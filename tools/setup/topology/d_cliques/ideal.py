#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
from random import Random
import setup.meta as m
import setup.nodes
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights
import interclique
import utils

def cliques(nodes, params):
    assert 'dataset' in params.keys(), "Missing 'dataset' params."
    assert 'nb-classes' in params['dataset'].keys(), \
           "Missing 'nb-classes' from 'dataset'."
    clique_size = params['dataset']['nb-classes']
    assert 'nodes' in params.keys(), "Missing 'nodes' params."
    nodes_per_class = params['nodes']['nodes-per-class']
    assert all(map(lambda x: x == nodes_per_class[0], nodes_per_class)),\
            "All classes should be equally represented."
    assert params['nodes']['local-classes'] == 1,\
            "Nodes should have examples of only one class."
    node0_size = setup.nodes.size(nodes[0])
    assert all(map(lambda x: setup.nodes.size(x) == node0_size, nodes)),\
            "All nodes should have the same number of examples."

    remaining = [ n["rank"] for n in nodes ]
    edges = {}

    dissimilarity = metrics.get('dissimilarity', params)

    def distance(n, clique):
        if len(clique) == 0:
            return 0
        return sum([ dissimilarity(nodes[n], nodes[m]) for m in clique ])

    cliques = []
    while len(remaining) > 0:
        clique = []
        for _ in range(clique_size):
            if len(remaining) == 0:
                break

            remaining.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(distance(r1, clique) - distance(r2, clique)))))
            n = remaining.pop()
            edges[n] = set(clique)
            for m in clique:
                edges[m] = { n } if not m in edges else edges[m].union({n})
            clique.append(n)
        cliques.append(clique)

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
    parser.add_argument('--remove-clique-edges', type=int, default=0, metavar='N', 
            help="Remove X random edges from each clique. ( default: 0)")
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    topology_params = {
        'name': 'd-cliques/ideal',
        'weights': args.weights,
        'interclique-topology': args.interclique,
        'remove-clique-edges': args.remove_clique_edges
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir) # reload extended

    cliques, intra_edges  = cliques(nodes, params)
    # In contrast to pseudo code of paper, the edge set is directly
    # extended by the interclique (interconnect) method. 'edges' therefore
    # both includes the intraclique and interclique edges.
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
