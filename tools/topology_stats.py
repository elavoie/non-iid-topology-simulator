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
        print("nb edges per node")
        print('    min: {}'.format(min(nb_edges)))
        print('    max: {}'.format(max(nb_edges)))
        print('    avg: {}'.format(sum(nb_edges)/len(nb_edges)))
        print()
