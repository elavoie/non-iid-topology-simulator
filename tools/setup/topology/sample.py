#!/usr/bin/env python
import argparse
import json
import logging
import os

import setup.meta as m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Used to indicate that we should use sampling.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    parser.add_argument('--sample-size', type=int, default=2,
      help="The number of nodes to be picked each round.")
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    params = {
        'name': 'sample',
        'sample-size': args.sample_size
    }
    m.extend(rundir, 'topology', params)

    topology = {
      'edges': {},
      'weights': [],
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
