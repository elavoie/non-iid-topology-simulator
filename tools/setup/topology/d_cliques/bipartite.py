#!/usr/bin/env python
import os
import argparse
import json
from random import Random

import numpy as np

import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights
import metrics
import interclique
import decentralized_algorithms.decentralized_greedy_bipartite_graph as dgb

import logging


def cliques(_nodes, _params):
    max_clique_size = _params['topology']['max-clique-size']
    # _nodes = [n['rank'] for n in _nodes]
    rand = Random()
    rand.seed(_params['meta']['seed'])

    rng = np.random.default_rng(_params['meta']['seed'])
    global_distribution = np.array(metrics.dist(_nodes))
    all_cliques = []
    for n in _nodes:
        all_cliques.append(
            dgb.DecentralizedClique(
                metrics.dist([n]), global_distribution, n['rank']))

    _cliques, _ = dgb.decentralized_greedy_bipartite_solution(
        all_cliques, rng, global_distribution, iterations=30,
        max_n_nodes=max_clique_size)

    cliques_as_lists = []
    for c in _cliques:
        cliques_as_lists.append(c.nodes_ids)

    _edges = {}
    for c in _cliques:
        ranks = set(c.nodes_ids)
        for rank in ranks:
            _edges[rank] = ranks.difference([rank])

    return cliques_as_lists, _edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a D-Cliques Topology.')
    parser.add_argument(
        '--rundir', type=str, default=None,
        help='Directory of the run in which to save the topology options.')
    parser.add_argument(
        '--interclique', type=str, default='fully-connected',
        choices=['ring', 'fractal', 'smallworld', 'fully-connected'],
        help="Interclique topology (default: 'fully-connected')")
    parser.add_argument(
        '--weights', type=str, default='metropolis-hasting',
        choices=['metropolis-hasting', 'equal-clique-probability'],
        help="Algorithm used to compute weights "
             "(default: 'metropolis-hasting').")
    parser.add_argument(
        '--max-clique-size', type=int, default=30, metavar='N',
        help='Maximum number of nodes in a clique (default: 30)')
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(
        level=getattr(logging, params['meta']['log'].upper(), None))

    topology_params = {
        'name': 'd-cliques/random-cliques',
        'weights': args.weights,
        'interclique-topology': args.interclique,
        'max-clique-size': args.max_clique_size
    }
    m.extend(rundir, 'topology', topology_params)

    cliques, intra_edges = cliques(nodes, m.params(rundir))
    # In contrast to pseudo code of paper, the edge set is directly
    # extended by the interclique (interconnect) method. 'edges' therefore
    # both includes the intraclique and interclique edges.
    logging.info(len(cliques))
    edges = interclique.get(args.interclique)(cliques, intra_edges, params)

    topology = {
        'edges': {rank: list(edges[rank]) for rank in edges},
        'weights': compute_weights(nodes, edges, topology_params),
        'cliques': cliques
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
