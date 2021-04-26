from __future__ import print_function
import os
import json
import time
import sys
import argparse
import math
import convergence
import scipy.optimize


experiments = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Results')
    parser.add_argument('results', type=str, default='./results',
                    help='directory of results')
    parser.add_argument('--learning-rate', type=float, default=None,
            help='show only results with this learning rate (default: False)')
    parser.add_argument('--batch-size',    type=int, default=None,
            help='show only results with this batch-size (default: False)')
    parser.add_argument('--dist-optimization', type=str, default=None,
            help='show only results with this distributed optimization algorithm (default: False)')
    parser.add_argument('--test-set',    type=str, default='valid', choices=['test', 'valid'],
            help='test set used (default: valid)')
    args = parser.parse_args()
    print(args)

    for entry in os.listdir(args.results):
        with open(os.path.join(args.results, entry, 'meta.json'), 'r') as meta_file:
            meta = json.load(meta_file)

        if args.learning_rate != None and meta['learning_rate'] != args.learning_rate:
            continue

        if args.batch_size != None and meta['batch_size'] != args.batch_size:
            continue

        if args.dist_optimization != None and meta['dist_optimization'] != args.dist_optimization:
            continue

        conv = convergence.from_dir(os.path.join(args.results, entry), args.test_set)

        experiments[entry] = {
           'timestamp': entry,
           'meta': meta,
           'convergence': conv
        }

    s = sorted([ experiments[entry] for entry in experiments ], key=lambda x: float(x['meta']['learning_rate']))
    s = sorted(s, key=lambda x: float(x['meta']['batch_size']))

    for entry in s:
        epochX = []
        for acc in [0.8, 0.85, 0.9, 0.91,0.92]:
            e = entry['convergence']['epochs'][acc]


            if e == None:
                epochX.append('*>999')
            else: 
                assert type(e) == int, "Invalid epoch value {} in {}".format(e, entry)
                epochX.append('{:5d}'.format(e) if e > 0 else '*{:4d}'.format(-e))
        print("{} {:6} bsz {:4d} lr {:.4f} 80\%:{} 85\%:{} 90\%:{} 91\%:{} 92\%:{} avg-neg-delta-avg:{:.4f}"
                .format(entry['timestamp'], 
                        entry['meta']['dist_optimization'],
                        entry['meta']['batch_size'], 
                        entry['meta']['learning_rate'], 
                        epochX[0],
                        epochX[1],
                        epochX[2],
                        epochX[3],
                        epochX[4],
                        entry['convergence']['avg-neg-delta-avg'])) 
