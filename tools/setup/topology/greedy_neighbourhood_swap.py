#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
from random import Random
import random_graph
import setup.meta as m
import setup.topology.d_cliques.metrics as metrics
from setup.topology.weights import compute_weights

def create(nodes, params):
    edges = random_graph.create(nodes, params)
    edges = { rank: set(edges[rank]) for rank in edges }

    seed = params['meta']['seed']
    rand = Random()
    rand.seed(seed)
    nb_neighbours = params['topology']['nb-neighbours']
    nb_passes = params['topology']['nb-passes']

    assert all([ len(edges[rank]) == nb_neighbours for rank in edges ]),\
        'Invalid starting random graph, expected {} edges per node'.format(nb_neighbours)

    assert all([ rank not in edges[rank] for rank in edges ]),\
        'Invalid starting random graph, exported node {} not to be part of its own edges {}'.format(nb_neighbours, rank, edges[rank])

    G_dist = metrics.dist(nodes)
    def skew(N):
        return metrics.skew(metrics.dist([ nodes[n] for n in N ]), G_dist)

    logging.info('initial average skew {}'.format(sum([ skew(edges[rank].union({rank})) for rank in edges ])/len(nodes)))

    for p in range(nb_passes): 
        for n in nodes:
            rank = n['rank']
            N = set([ nn for nn in edges[rank] ])
            m_rank = rand.sample(N, 1)[0]
            M = set([ mn for mn in edges[m_rank] ]) 

            current_skew = skew(N.union({rank})) + skew(M.union({m_rank}))
            candidates = []
            for x in rand.sample(N, len(N)):
                for y in rand.sample(M, len(M)):
                    new_skew = skew(N.difference({x}).union({y,rank})) +\
                               skew(M.difference({y}).union({x,m_rank}))
                    if new_skew < current_skew and (x not in M) and (y not in N):
                        #logging.info('new skew {} previous best skew {} swap ({},{}) '.format(new_skew, best_skew, x, y))
                        candidates.append((x,y))
            
            if len(candidates) > 0:
                best_x,best_y = rand.sample(candidates, 1)[0]
                assert len(edges[rank]) == nb_neighbours
                assert len(edges[m_rank]) == nb_neighbours
                #logging.info('swapping ({},{}) edges between nodes {} and {}'.format(best_x,best_y,rank, m_rank))

                edges[rank].remove(best_x)
                edges[rank].add(best_y)
                edges[m_rank].remove(best_y)
                edges[m_rank].add(best_x)

                assert len(edges[rank]) == nb_neighbours
                assert len(edges[m_rank]) == nb_neighbours

                #logging.info('pass {} node {} skew: {}'.format(p, rank, skew(edges[rank].union({rank}))))
                #logging.info('pass {} node {} skew: {}'.format(p, m_rank, skew(edges[m_rank].union({m_rank}))))
    
    skews = [ skew(edges[rank].union({rank})) for rank in edges ]
    logging.info('final skew min {:.3f} max {:.3f} avg {:.3f}'.format(min(skews), max(skews), sum(skews)/len(nodes)))

    return { rank: list(edges[rank]) for rank in edges }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Random Graph Topology with ' +\
                                                 'neighbhourhoods for each node with lower skew.' +\
                                                 'Analogous to D-Cliques Greedy Swap topology.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--weights', type=str, default='metropolis-hasting', 
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    parser.add_argument('--nb-neighbours', type=int, default=10,
      help='Number of neighbours for each node. (default: 10)')
    parser.add_argument('--nb-passes', type=int, default=None,
      help='Number of passes over the node set used to find ' + 
           'candidates for swapping. (default: same as --nb-neighbours)')
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    if args.nb_passes is None:
        args.nb_passes = args.nb_neighbours

    topology_params = {
        'name': 'greedy-neighbourhood-swap',
        'nb-neighbours': args.nb_neighbours,
        'nb-passes': args.nb_passes,
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', topology_params)
    params = m.params(rundir)

    edges = create(nodes, params)
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, topology_params),
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
