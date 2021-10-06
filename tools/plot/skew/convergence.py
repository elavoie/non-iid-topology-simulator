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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show the convergence speed of skew over many experiments.')
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
    parser.add_argument('--linewidth', type=float, default=1.5,
                    help='Line width of plot lines.')
    parser.add_argument('--max-steps', type=int, default=1000,
                    help='Maximum number of steps to use on X axis.')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    if args.rundirs is None:
        stdin = sys.stdin.readlines()
        rundirs = [ line.replace('\n','') for line in stdin ]
    else: 
        rundirs = args.rundirs

    seeds = {}
    convergences = {}
    for rundir in rundirs:
        params = m.params(rundir)

        h = hash(params)
        s = m.params(rundir, 'meta')['seed']
        if h not in seeds.keys():
            seeds[h] = set()
            convergences[h] = {}
        assert s not in seeds[h], "Seed {} already seen previously for {}".format(s, h)
        seeds[h].add(s)
        convergences[h][rundir] = skew_convergence(rundir)

    if len(args.labels) == 0:
        labels = [ 'undefined' for _ in convergences.keys() ]
    elif len(args.labels) < len(convergences.keys()):
        print('Insufficient number of labels')
        sys.exit(1)
    else:
        labels = args.labels

    if len(args.linestyles) == 0:
        linestyles = [ '-' for _ in convergences.keys() ]
    elif len(args.linestyles) < len(convergences.keys()):
        print('Insufficient number of linestyles')
        sys.exit(1)
    else:
        linestyles = args.linestyles

    fig, ax = plt.subplots()
    ax.set_ylabel('Skew', fontsize=args.font_size)
    ax.set_xlabel('Steps', fontsize=args.font_size)
    ax.tick_params(labelsize=args.font_size)

    for h,lb,ls in zip(convergences, labels, linestyles):
        logging.info('creating curves for {}'.format(h))
        last = {
            rundir: {
                "min": 0,
                "max": 0,
                "avg": 0
            } for rundir in convergences[h]
        }
        minuses = []
        maxes = []
        avgs = []
        for k_int in range(args.max_steps):
            k = str(k_int)
            logging.info('computing stats for step {}'.format(k))
            k_minuses = []
            k_maxes = []
            k_avgs = []
            for rundir in convergences[h]:
                logging.info('generating data for {}'.format(rundir))
                c = convergences[h][rundir]['convergence']
                if k in c.keys():
                    k_minuses.append(c[k]['min'])
                    k_maxes.append(c[k]['max'])
                    k_avgs.append(c[k]['avg'])
                    last[rundir]['min'] = c[k]['min']
                    last[rundir]['max'] = c[k]['max']
                    last[rundir]['avg'] = c[k]['avg']
                else:
                    k_minuses.append(last[rundir]['min'])
                    k_maxes.append(last[rundir]['max'])
                    k_avgs.append(last[rundir]['avg'])
            minuses.append(min(k_minuses))
            maxes.append(max(k_maxes))
            avgs.append(sum(k_avgs) / len(k_avgs))

        print(h)
        avg=plt.plot(range(args.max_steps), avgs, label=lb, linestyle=ls, linewidth=args.linewidth)
        plt.plot(range(args.max_steps), minuses, linestyle=ls, linewidth=args.linewidth/2., color=avg[0].get_color())
        plt.plot(range(args.max_steps), maxes, linestyle=ls, linewidth=args.linewidth/2., color=avg[0].get_color())

    matplotlib.rc('font', size=args.font_size)
    plt.legend()

    if args.save_figure is not None:
        plt.savefig(args.save_figure, transparent=True, bbox_inches='tight')

    plt.show() 



