#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
from random import Random
import setup.nodes
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights

def create(nodes, params):
    ## Assert that
    # 1. All nodes have examples of only a single class
    # 2. That all classes are equally represented
    # 3. All nodes have the same number of examples
    assert 'dataset' in params.keys(), "Missing 'dataset' params."
    assert 'nb-classes' in params['dataset'].keys(), \
           "Missing 'nb-classes' from 'dataset'."
    nb_classes = params['dataset']['nb-classes']
    assert 'nodes' in params.keys(), "Missing 'nodes' params."
    nodes_per_class = params['nodes']['nodes-per-class']
    assert all(map(lambda x: x == nodes_per_class[0], nodes_per_class)),\
            "All classes should be equally represented."
    assert params['nodes']['local-classes'] == 1,\
            "Nodes should have examples of only one class."
    node0_size = setup.nodes.size(nodes[0])
    assert all(map(lambda x: setup.nodes.size(x) == node0_size, nodes)),\
            "All nodes should have the same number of examples."

    metric = metrics.dissimilarity
    nb_classes = params['dataset']['nb-classes']

    # Even if neighbours don't necessarily have edges among themselves,
    # this distance metric, when used with 'dissimilarity' ensures a 
    # representation of all classes.
    def distance(n, neighbours):
        if len(neighbours) == 0:
            return 0
        return sum([ metric(nodes[n], nodes[m]) for m in neighbours ])

    def class_in_neighbours(c, neighbours):
        for n in neighbours:
            if any([ int(c1) & int(c2) for c1,c2 in zip(c, nodes[n]['classes']) ]):
                return True

    edges = { n['rank']:set() for n in nodes }

    for n in nodes:
        rank = n['rank']
        available = [ m['rank'] for m in nodes 
                      if m['rank'] != rank
                      and len(edges[m['rank']]) < nb_classes
                      and m['rank'] not in edges[rank] 
                      and not class_in_neighbours(n['classes'], edges[m['rank']])]

        neighbours = edges[rank].union({rank})
        toadd = (nb_classes - 1) - len(edges[rank])
        for _ in range(toadd):
            available.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(distance(r1, neighbours) - distance(r2, neighbours)))))
            new_neighbour = available.pop()
            neighbours.add(new_neighbour)
            edges[rank].add(new_neighbour)
            edges[new_neighbour].add(rank)

    neighbourhoods = { n['rank']:list(edges[n['rank']].union({n['rank']})) for n in nodes }
    
    # Add one last edge to all nodes,
    # this last edge does not participate in neighbourhood averaging
    seed = params['meta']['seed']
    rand = Random()
    rand.seed(seed)
    for n in nodes:
        rank = n['rank']
        available = [ m['rank'] for m in nodes 
                      if m['rank'] != rank
                      and len(edges[m['rank']]) < nb_classes
                      and m['rank'] not in edges[rank] ]
        rand.shuffle(available)
        toadd = (nb_classes - len(edges[rank]))
        for neighbour in available[:toadd]:
            edges[rank].add(neighbour)
            edges[neighbour].add(rank)

    for n in nodes:
        rank = n['rank']
        assert len(edges[rank]) == params['topology']['nb-neighbours']
        classes = n['classes'].copy()
        for m in edges[n['rank']]:
            for c in range(nb_classes):
                classes[c] += nodes[m]['classes'][c]
        for c in range(nb_classes):
            assert classes[c] >= 1 and classes[c] <= 2

    return { rank: list(edges[rank]) for rank in edges }, neighbourhoods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Random Topology in which all classes '+\
      'are represented in the neighbourhood of a node.')
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
        'name': 'greedy-diverse-random',
        'nb-neighbours': args.nb_neighbours,
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir)

    edges, neighbourhoods = create(nodes, params)
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, topology_params),
      'neighbourhoods': neighbourhoods
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
