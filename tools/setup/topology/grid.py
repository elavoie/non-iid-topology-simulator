#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import math
import setup.meta as m
import setup.topology.metrics as metrics
from setup.topology.weights import compute_weights

# generate a grid by placing nodes in a clockwise spiral
# from an initial position
# -> enables choosing the initial position of a single node
# -> favors clusters in a box fashion (rather than diagonal)
# 
# Grid Coordinates (i,j): 
# (0,0) -> (1,0) -> ...
#   |
#   V
# (0,1)
#   | 
#   V
#  ...
def create(nodes, metric):
    def out_of_bounds (i,j):
        return i < 0 or i >= side_len or j < 0 or j >= side_len

    def distance (i,j,node):
        d = 0.0
        d += metric(grid[i-1][j], node) if not out_of_bounds(i-1, j) and grid[i-1][j] != None else 0.0
        d += metric(grid[i+1][j], node) if not out_of_bounds(i+1, j) and grid[i+1][j] != None else 0.0
        d += metric(grid[i][j-1], node) if not out_of_bounds(i, j-1) and grid[i][j-1] != None else 0.0
        d += metric(grid[i][j+1], node) if not out_of_bounds(i, j+1) and grid[i][j+1] != None else 0.0
        return d

    position = (0,0)

    side_len = math.ceil(math.sqrt(len(nodes)))
    assert side_len == math.sqrt(len(nodes)), "grid: unsupported non-square node number"

    # Current box limits
    left = position[0]
    right = position[0]
    top = position[1]
    bottom = position[1]
    # Current coordinates
    i = position[0]
    j = position[1]
    # Remaining nodes
    remaining = [ i for i in range(len(nodes)) ]
    # Uninitialized Grid
    grid = [ [ None for j in range(side_len) ] for i in range(side_len) ]

    # Place first node
    node = nodes[remaining.pop()]
    grid[i][j] = node

    # Topology
    edges = {node["rank"]: set()}

    # Set base case
    last_i = i
    last_j = j
    last_position = 'right-top'

    while len(remaining) > 0:
        # increase box limits once the box is filled
        if i == last_i and j == last_j and last_position == 'right-top':
            right += 1
            bottom += 1
            last_position = 'left-bottom'
            last_i = left
            last_j = bottom
        if i == last_i and j == last_j and last_position == 'left-bottom':
            left -= 1
            top -= 1
            last_position = 'right-top'
            last_i = right
            last_j = top

        # pick next empty spot in a spiral pattern
        if i < right and j == top:
            i += 1
        elif j < bottom and i == right:
            j += 1
        elif i > left and j == bottom:
            i -= 1
        elif j > top and i == left:
            j -= 1

        # skip spots outside of grid limits
        if out_of_bounds(i,j):
            continue

        # find best next node, according to the metric evaluated
        # with previously assigned neighbours
        distances = { rank:distance(i,j,nodes[rank]) for rank in remaining }
        remaining.sort(key=functools.cmp_to_key(lambda a, b: distances[a] - distances[b]))

        # place node
        rank = remaining.pop() # best node
        grid[i][j] = nodes[rank]

        # assign edges
        for (ii,jj) in [(i,j-1), (i,j+1), (i+1,j), (i-1,j)]:
            if not out_of_bounds(ii, jj) and grid[ii][jj] != None:
                neighbour = grid[ii][jj]["rank"]
                edges[rank] = edges[rank].union({neighbour}) if rank in edges else {neighbour}
                edges[neighbour] = edges[neighbour].union({rank}) if neighbour in edges else {rank} 

    return { rank: list(edges[rank]) for rank in edges }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Grid Topology.')
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
        'name': 'grid',
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
