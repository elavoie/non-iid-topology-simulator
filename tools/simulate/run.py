#!/usr/bin/env python
from importlib import import_module
import argparse
import setup.meta as m
import logging
import torch
import torchvision
import os
from random import Random
import setup.dataset as dataset
import setup.topology as t
import simulate.logger as logger
from torch.multiprocessing import Process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate the pre-configured run.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run from which to load options.')
    parser.add_argument('--nb-epochs', type=int, default=10, metavar='N',
                        help='number of epochs (default: 10)')

    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    algo = import_module(params['algorithm']['module'])
    event_dir = os.path.join(rundir, 'events')
    node_desc = m.load(rundir, 'nodes.json')

    m.extend(rundir, 'simulator', {
      'nb-epochs': args.nb_epochs
    })

    # Logger initialized here so logging processes are siblings of the main
    # process. Avoids deadlock on Linux (see below).
    logging.info('Starting logger')
    log = logger.init(params, rundir)

    def run(log, rundir, params):
        seed = params['meta']['seed']
        torch.manual_seed(seed)         
        torchvision.utils.torch.manual_seed(seed)
        model = import_module(params['model']['module'])
        train = dataset.train(params)
        train_ind, val_ind = dataset.partition([ n['samples'] for n in node_desc ], params)

        nodes = []
        for rank in range(params['nodes']['nb-nodes']):
            logging.info('creating node {}'.format(rank))
            _model = model.create(params)
            optimizer = algo.optimizer(_model, params)

            # Create empty event-file first to avoid race condition on first write
            event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
            open(event_file, 'a').close()

            nodes.append({
              'rank': rank,
              'event-file': event_file,
              'train-set': [ train[i] for i in train_ind[rank] ],
              'train-iterator': None,
              'model': _model,
              'optimizer': optimizer
            })

            # Initialize topology
            topology = t.load(rundir)

        state,loss,done = algo.init(nodes, topology, params)
        log.state(params, 0, state)
        for epoch in range(1,args.nb_epochs+1):
            print('epoch {}'.format(epoch))
            while not done:
                state,loss,done = algo.next(state, params)
                if not done:
                    log.loss(loss)
            log.state(params, epoch, state)
            done = False
        logging.info('run() done')

    # The main loop is also run in a separate process to avoid the deadlock
    # on Linux when the MNIST dataset is open both in a parent process
    # (this one) and child processes (the log_models) created later. This
    # issue still happens when the dataset is saved (torch.save) and
    # reloaded later (torch.load). When all processes are siblings this is
    # no longer an issue.
    logging.info('Starting main process')
    main = Process(target=run, args=(log, rundir, params))
    main.start()
    main.join()

    logging.info('Stopping Logger')
    log.stop()
