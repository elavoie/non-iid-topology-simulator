#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
from random import Random
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights

def create(nodes, params):
    seed = params['meta']['seed']
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Random Graph Topology.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--weights', type=str, default='metropolis-hasting', 
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    parser.add_argument('--nb-neighbours', type=int, default=10,
      help='Number of neighbours for each node.')
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    topology_params = {
        'name': 'random-graph',
        'nb-neighbours': args.nb_neighbours,
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir)

    edges = create(nodes, params)
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, params),
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
