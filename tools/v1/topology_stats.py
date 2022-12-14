import argparse
import os
import json
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print stats about topology')
    parser.add_argument('results', type=str, nargs='+', default=[], help='experiment result')

    args = parser.parse_args()

    for result in args.results:
        print(result) 
        topology = json.load(open(result + '/topology.json'))
        nb_edges = [ len(topology[n]) for n in topology.keys() ]
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
