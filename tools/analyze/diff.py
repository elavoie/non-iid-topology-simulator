#!/usr/bin/env python
import argparse
import os
import json
import sys
import setup.meta as m
import logging
import properties

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List differing properties between rundirs.')
    parser.add_argument('--rundirs', type=str, nargs='+', default=None,
            help='experiment rundirs')
    parser.add_argument('--log', type=str, choices=['ERROR', 'INFO', 'DEBUG', 'WARNING'], 
            default='WARNING', help='Logging level.')
    parser.add_argument('--only', type=str, nargs='+', default=[], 
            help='Show differences only for those properties.')



    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    if args.rundirs is None:
        stdin = sys.stdin.readlines()
        rundirs = [ line.replace('\n','') for line in stdin ]
    else: 
        rundirs = args.rundirs

    all_params = [ m.params(rundir) for rundir in rundirs ]

    common = []
    different = []
    for p in properties.properties:
        name = ':'.join(p[0])

        values = [ properties.get(params, p[0]) for params in all_params ]
        if all([ v == values[0] for v in values]):
            common.append((name, values[0]))
        else:
            different.append((name, values))

    print('Identical parameters')
    print('--------------------')
    max_len_p = max([len(p) for (p,_) in common])
    for (p,v) in common:
        print(('{:' + str(max_len_p) + '} {}').format(p, v))
    print()

    print('Differing parameters')
    print('--------------------')
    if len(different) > 0:
        max_len_p = [ max([ len(rundir) for rundir in rundirs ]) ] + [ max([ len(str(i)) for i in vs ] + [len(p)]) for (p,vs) in different ]
        f_str = " ".join([ "{:" + str(l) + "}" for l in max_len_p ])
        print(f_str.format(*tuple(['rundir'] + [ p for (p,_) in different ])))
        for rundir, i in zip(rundirs, range(len(all_params))):
            print(f_str.format(*tuple([rundir] + [ str(v[i]) for (p,v) in different ])))
    else:
        print('None')
    print('')





