#!/usr/bin/env python
import os
import sys
import json
import time
import argparse
import logging
import socket
from subprocess import check_output

def rundir(args):
    assert hasattr(args, 'rundir'), "Invalid args inputs, " +\
      "should have 'rundir' attribute set by ArgumentParser"

    if args.rundir is None:
        stdin = sys.stdin.readlines()
        assert len(stdin) >= 1 , "Invalid standard output from previous process: expected RUNDIR on first line."
        rundir = stdin[0].split('\n')[0]
    else: 
        rundir = args.rundir
    assert os.path.exists(rundir), "Invalid run directory '{}'".format(rundir)
    return rundir

def params(rundir, name=None):
    path = os.path.join(rundir, 'params.json')
    if not os.path.exists(path):
        params = {} 
    else:
        with open(path, 'r+') as param_file:
            params = json.load(param_file)

    if name is None:
        return params
    else:
        assert name in params.keys(), "Invalid property name {} for params.json object".format(name)
        return params[name]

def load(rundir, filename):
    path = os.path.join(rundir, filename)
    with open(path, 'r+') as f:
        obj = json.load(f)
    return obj 

def extend(rundir, name, _dict):
    p = params(rundir)

    assert name not in p.keys(), "Cannot extend params.json with {}, property already exists.".format(name)
    p[name] = _dict

    with open(os.path.join(rundir, 'params.json'), 'w+') as param_file:
        json.dump(p, param_file, indent=4)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup an Experiment Run')
    parser.add_argument('--seed', type=int, default=1337, metavar='N',
        help='seed for pseudo-random number generator')
    parser.add_argument('--log', type=str, default='WARNING', 
        choices=['NOTSET', 'INFO', 'DEBUG', 'PARTITION', 'WARNING'],
        help='log level to use (default: WARNING)')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    parser.add_argument('--results-directory', type=str, default=results_dir,
        help='directory in which to save the run. (default: {})'.format(results_dir))

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    # Setup Results Directory
    hostname = socket.gethostname()
    rundir = os.path.join(args.results_directory, time.strftime('%Y-%m-%d-%H:%M:%S-%Z') + '-{}'.format(hostname))
    os.makedirs(rundir)
    event_dir = os.path.join(rundir, 'events')
    os.makedirs(event_dir)

    # Save Experiment Parameters and Script Version
    meta = {
        "seed": args.seed,
        "log": args.log,
        "results-directory": args.results_directory,
        "script": __file__,
        "git-hash": check_output(['git', 'rev-parse', '--short', 'HEAD'])[:-1].decode()
    }

    extend(rundir, 'meta', meta)
    print(rundir)
