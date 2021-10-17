#!/usr/bin/env python
import argparse
import os
import json
import sys
import setup.meta as m
import setup.nodes as ns
import setup.topology as topology
import setup.topology.d_cliques.metrics as dc_metrics
import logging
import analyze.properties as properties

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def hash(params):
    kv = []
    for p in properties.properties:
        name = ':'.join(p[0])
        if name == 'meta:seed':
            continue
        kv.append(name + '=' + str(properties.get(params, p[0])))
    return ';'.join(kv)

def skew_convergence(rundir):
    global_events = os.path.join(rundir, 'events', 'global.jsonlines')
    if not os.path.exists(global_events):
        return None

    skews = None
    with open(global_events, 'r') as global_file:
        for line in global_file:
            event = json.loads(line) 
            if event['type'] == 'skew-convergence':
                assert skews is None, "Expected only a single skew-convergence event in events/global.jsonlines"
                skews = event
    return skews

def avg_final_skew(rundir):
    sc = skew_convergence(rundir)
    if sc is None:
        nodes = ns.load(rundir)
        cliques = topology.load(rundir)['cliques']
        global_dist = dc_metrics.dist(nodes)
        skews = [ dc_metrics.skew(global_dist, dc_metrics.dist([ nodes[r] for r in c ])) for c in cliques ]
        return sum(skews)/len(skews), 0.0
    else:
        convergence = sc['convergence']
        max_step = 0
        final = None
        for k in convergence.keys():
            step = int(k)
            if step > max_step:
                final = convergence[k]
        assert final is not None, "Final skews not found"
        return final['avg'], sc['duration']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show the distribution of final skews of many experiments.')
    parser.add_argument('--rundirs', type=str, nargs='+', default=None,
            help='experiment rundirs')
    parser.add_argument('--log', type=str, choices=['ERROR', 'INFO', 'DEBUG', 'WARNING'], 
            default='WARNING', help='Logging level.')
    parser.add_argument('--labels', type=str, nargs='+', default=[],
                    help='Labels that will appear in the legend')
    parser.add_argument('--linestyles', type=str, nargs='+', default=[],
                    help='Linestyles used for each curve')
    parser.add_argument('--legend', type=str, default='best',
                    help='Position of legend (default: best).')
    parser.add_argument('--font-size', type=int, default=16,
                    help='Font size (default: 16).')
    parser.add_argument('--save-figure', type=str, default=None,
                    help='File in which to save figure.')
    parser.add_argument('--xmin', type=float, default=0.,
                    help='Minimum value on the x axis.')
    parser.add_argument('--xmax', type=float, default=0.75,
                    help='Maximum value on the x axis.')
    parser.add_argument('--nb-bins', type=int, default=100,
                    help='Number of bins of histogram.')
    parser.add_argument('--linewidth', type=float, default=1.5,
                    help='Line width of plot lines.')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    if args.rundirs is None:
        stdin = sys.stdin.readlines()
        rundirs = [ line.replace('\n','') for line in stdin ]
    else: 
        rundirs = args.rundirs

    seeds = {}
    avg_skews = {}
    durations = {}
    for rundir in rundirs:
        params = m.params(rundir)
        h = hash(params)

        if h not in avg_skews.keys():
            seeds[h] = set({})
            avg_skews[h] = []
            durations[h] = []
        
        s = m.params(rundir, 'meta')['seed']
        assert s not in seeds[h], "Seed {} already seen previousl for {}".format(s, h)
        seeds[h].add(s)
        avg, t = avg_final_skew(rundir)
        avg_skews[h].append(avg)
        durations[h].append(t)

    if len(args.labels) == 0:
        labels = [ 'undefined' for _ in avg_skews.keys() ]
    elif len(args.labels) < len(avg_skews.keys()):
        print('Insufficient number of labels')
        sys.exit(1)
    else:
        labels = args.labels

    if len(args.linestyles) == 0:
        linestyles = [ '-' for _ in avg_skews.keys() ]
    elif len(args.linestyles) < len(avg_skews.keys()):
        print('Insufficient number of linestyles')
        sys.exit(1)
    else:
        linestyles = args.linestyles

    freqs = [ len(avg_skews[h]) for h in avg_skews ]
    assert all(map(lambda x: x == freqs[0], freqs)), "Inconsistent number of experiments"

    fig, ax = plt.subplots()
    ax.set_ylabel('Frequency over {}'.format(freqs[0]), fontsize=args.font_size)
    ax.set_xlabel('Skew', fontsize=args.font_size)
    ax.tick_params(labelsize=args.font_size)

    for h,l,ls in zip(avg_skews.keys(), labels, linestyles):
        print(h)
        print('min duration: {}'.format(min(durations[h])))
        print('avg duration: {}'.format(sum(durations[h])/len(durations[h])))
        print('max duration: {}'.format(max(durations[h])))
        print('final avg skew: {}'.format(sum(avg_skews[h])/len(avg_skews[h])))
        print()
        plt.hist(avg_skews[h], bins=np.linspace(args.xmin,args.xmax,args.nb_bins), histtype='step', label=l, linestyle=ls, linewidth=args.linewidth)

    matplotlib.rc('font', size=args.font_size)
    plt.legend()

    if args.save_figure is not None:
        plt.savefig(args.save_figure, transparent=True, bbox_inches='tight')

    plt.show() 



