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
    parser.add_argument('--sample-method', type=str, default="random",
      help='The method used for sample (random, random-with-overlap).')
    parser.add_argument('--sample-overlap', type=int, default=0,
      help='The overlap between subsequent samples. Only used when using the "random with overlap" sampling method.')
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    nodes = m.load(rundir, 'nodes.json')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    assert args.sample_overlap <= args.sample_size, "The sample overlap must be less than or equal to the sample size."

    params = {
        'name': 'sample',
        'sample-size': args.sample_size,
        'sample-method': args.sample_method,
        'sample-overlap': args.sample_overlap
    }
    m.extend(rundir, 'topology', params)

    topology = {
      'edges': {},
      'weights': [],
    }
    with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
        json.dump(topology, topology_file)

    print(rundir)
