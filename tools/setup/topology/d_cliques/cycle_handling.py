#!/usr/bin/env python
import os
import argparse
import logging
import json
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights
import interclique
import decentralized_algorithms.decentralized_greedy_resolving_conflicts as \
    cycle_handling
import metrics
import numpy as np


def cliques(nodes, params):
    assert 'dataset' in params.keys(), "Missing 'dataset' params."
    assert 'nb-classes' in params['dataset'].keys(), \
           "Missing 'nb-classes' from 'dataset'."
    assert 'nodes' in params.keys(), "Missing 'nodes' params."
    max_clique_size = params['topology']['max-clique-size']

    all_cliques = []
    global_distribution = metrics.dist(nodes)
    for i in range(len(nodes)):
        all_cliques.append(cycle_handling.DecentralizedClique(metrics.dist(
            [nodes[i]]), global_distribution, nodes[i]["rank"]))

    decentralized_greedy_parameters = {"all_cliques": all_cliques,
                                       "rng": np.random.default_rng(
                                           params["meta"]["seed"]),
                                       "global_distribution": global_distribution,
                                       "iterations": 30,
                                       "random_sample": 10,
                                       "max_n_nodes": max_clique_size}
    all_cliques, _ = cycle_handling.decentralized_greedy_resolving_conflicts(
        **decentralized_greedy_parameters)

    edges = {}
    for c in all_cliques:
        ranks = set(c.nodes_ids)
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
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    topology_params = {
        'name': 'd-cliques/ideal',
        'weights': args.weights,
        'interclique-topology': args.interclique
    }
    m.extend(rundir, 'topology', topology_params)

    cliques, intra_edges = cliques(nodes, params)
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
