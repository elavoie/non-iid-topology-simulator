#!/usr/bin/env python
import argparse
import os
import json
import sys
import setup.meta as m
import logging

properties = [
  (['meta', 'seed'], int, None),
  (['dataset', 'name'], str, None),
  (['nodes', 'name'], str, None),
  (['nodes', 'nb-nodes'], int, None),
  (['topology', 'name'], str, None),
  (['topology', 'clique-gradient'], bool, None),
  (['topology', 'interclique'], str, None),
  (['model', 'name'], str, None),
]

def get(params, path):
    obj = params
    for x in path:
        if not x in obj.keys():
            return None
        else:
            obj = obj[x]
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select relevant experiments given expected properties.')
    parser.add_argument('results', type=str, help='Directory of experiment results.')
    parser.add_argument('--log', type=str, choices=['ERROR', 'INFO', 'DEBUG', 'WARNING'], default='WARNING', help='Logging level.')
    parser.add_argument('--inline', action='store_const', const=True, default=False, help='Print answers on a single line. (default: False)')

    for p in properties:
        parser.add_argument('--' + ':'.join(p[0]), type=p[1], default=p[2], help='')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    collected = []
    for path in os.listdir(args.results):
        rundir = os.path.join(args.results, path)
        if not os.path.isdir(rundir):
            continue

        logging.debug('checking {}'.format(rundir))
        params = m.params(rundir)
        keep = True
        for p in properties:
            name = ':'.join(p[0])
            arg_value = getattr(args, name.replace('-','_'))
            params_value = get(params, p[0])
            logging.debug('    {} expected {} got {}'.format(name, arg_value, params_value))

            if arg_value is None:
                continue

            if params_value is None or params_value != arg_value:
                keep = False
                break

        if keep:
            collected.append(rundir)

    collected = sorted(collected)
    if args.inline:
        print(' '.join(collected))
    else: 
        for rundir in collected:
            print(rundir)



