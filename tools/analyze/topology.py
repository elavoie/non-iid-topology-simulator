#!/usr/bin/env python
import argparse
import os
import json
import sys
import setup.topology as t
import setup.meta as m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print stats about topology')
    parser.add_argument('results', type=str, nargs='+', default=[], help='experiment result')

    args = parser.parse_args()

    for result in args.results:
        print(result) 
        params = m.params(result)
        topology_params = params['topology']
        print(topology_params)
        print()

        topology = t.load(result)
        edges = topology['edges']
        nb_edges = [ len(edges[n]) for n in edges.keys() ]
        distribution = {}
        for e in range(min(nb_edges), max(nb_edges)+1):
            nb = sum(map(lambda x: 1 if x == e else 0, nb_edges))
            if nb > 0:
                distribution[e] = nb

        print("nb edges per node")
        print('    min: {}'.format(min(nb_edges)))
        print('    max: {}'.format(max(nb_edges)))
        print('    avg: {}'.format(sum(nb_edges)/len(nb_edges)))
        print('    distribution:') 
        print('        (nb edges): (nb nodes)')
        for e in distribution.keys():
            print('        {}: {}'.format(e, distribution[e]))
        print()
        print("total")
        print('    edges: {}'.format(sum(nb_edges)))
        print('    nodes: {}'.format(len(nb_edges)))
        print()

        if 'cliques' in topology.keys():
            print("cliques")
            if 'max-clique-size' in topology_params:
                print("    max-clique-size: {}".format(topology_params['max-clique-size']))
            cliques = topology['cliques']
            min_clique_size = min([ len(c) for c in cliques ])
            max_clique_size = max([ len(c) for c in cliques ])
            print("    nb: {}".format(len(cliques)))
            print("    min: {}".format(min_clique_size))
            print("    max: {}".format(max_clique_size))
            print("    avg: {}".format(sum([ len(c) for c in cliques ])/len(cliques)))

            distribution = {}
            for s in range(min_clique_size, max_clique_size+1):
                nb = sum(map(lambda c: 1 if len(c) == s else 0, cliques))
                if nb > 0:
                    distribution[s] = nb

            print('    distribution:') 
            print('        (nb nodes): (nb cliques)')
            for s in distribution.keys():
                print('        {}: {}'.format(s, distribution[s]))
