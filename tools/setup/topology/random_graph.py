#!/usr/bin/env python
import os
import argparse
import logging
import json
from random import Random
import setup.meta as m
from setup.topology.weights import compute_weights

def create(nodes, params):
    seed = params['topology']['topology-seed']
    rand = Random()
    rand.seed(seed)
    nb_neighbours = params['topology']['nb-neighbours']

    found = False
    count = 0
    while not found:
        count += 1
        edges = { n['rank']: set() for n in nodes }

        for n in nodes:
            rank = n['rank']
            available = [ m['rank'] for m in nodes
                          if m['rank'] != rank
                          and len(edges[m['rank']]) < nb_neighbours
                          and m['rank'] not in edges[rank] ]
            rand.shuffle(available)
            toadd = (nb_neighbours - len(edges[rank]))
            for neighbour in available[:toadd]:
                edges[rank].add(neighbour)
                edges[neighbour].add(rank)

        found = True
        for n in nodes:
            if len(edges[n['rank']]) != nb_neighbours:
                found = False
                break
        if not found:
            logging.info('random_ten: current solution invalid, trying another one')
        assert count < 1000, "random_graph: could not find a working solution, aborting"
    return { rank: list(edges[rank]) for rank in edges }


def generate_topology(nodes, params):
    edges = create(nodes, params)
    topology = {
        'edges': edges,
        'weights': compute_weights(nodes, edges, params["topology"]),
    }
    return topology


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Random Graph Topology.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--weights', type=str, default='metropolis-hasting',
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    parser.add_argument('--nb-neighbours', type=int, default=10,
      help='Number of neighbours for each node.')
    parser.add_argument('--randomize', action='store_const', const=True, default=False,
      help="Whether to randomize the connections after every step. ( default: False)")
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    topology_params = {
        'name': 'random-graph',
        'nb-neighbours': args.nb_neighbours,
        'weights': args.weights,
        'randomize': args.randomize,
        'topology-seed': meta['seed']
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir)

    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(generate_topology(nodes, params), topology_file)

    print(rundir)
