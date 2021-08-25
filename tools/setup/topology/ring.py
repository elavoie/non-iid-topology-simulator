#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights

def create(nodes, metric):
    remaining = [ n["rank"] for n in nodes ]
    n = remaining.pop()
    edges = {n:set()}
    first = n
    while len(remaining) > 0:
        remaining.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(metric(nodes[r1], nodes[n]) - metric(nodes[r2], nodes[n])))))
        t = remaining.pop()
        edges[n] = { t } if not n in edges else edges[n].union({t})
        edges[t] = { n } if not t in edges else edges[t].union({n})
        n = t
    # complete the ring
    if (len(nodes) > 1):
        edges[n] = edges[n].union({first})
        edges[first] = edges[first].union({n})
    return { rank: list(edges[rank]) for rank in edges }

def load(rundir):
    topology = m.load(rundir, 'topology.json')
    edges = topology['edges']
    topology['edges'] = { int(rank):edges[rank] for rank in edges }
    topology['weights'] = torch.tensor(topology['weights'])
    return topology



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Ring Topology.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--metric', type=str, default='dissimilarity', 
      choices=['similarity', 'dissimilarity', 'random'],
      help='Metric used to place nodes.')
    parser.add_argument('--weights', type=str, default='metropolis-hasting', 
      choices=['metropolis-hasting', 'equal-clique-probability'],
      help="Algorithm used to compute weights (default: 'metropolis-hasting').")
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    params = {
        'name': 'ring',
        'metric': args.metric,
        'weights': args.weights 
    }
    m.extend(rundir, 'topology', params)

    edges = create(nodes, metrics.get(args.metric, meta))
    topology = {
      'edges': edges,
      'weights': compute_weights(nodes, edges, params),
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
