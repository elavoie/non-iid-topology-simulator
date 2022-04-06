#!/usr/bin/env python
from importlib import import_module
import argparse
import setup.meta as m
import logging
import torch
import torchvision
import os
import setup.dataset as dataset
import setup.topology as t
import simulate.logger as logger
from torch.multiprocessing import Process


def should_log(node, epoch_done, params, state):
    if params['logger']['accuracy-logging-interval'] and epoch_done and node['epoch'] % params['logger']['accuracy-logging-interval'] == 0:
        return True
    elif params['logger']['accuracy-logging-interval-steps'] and state['step'] % params['logger']['accuracy-logging-interval-steps'] == 0:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate the pre-configured run.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run from which to load options.')
    parser.add_argument('--nb-epochs', type=int, default=10, metavar='N',
                        help='number of epochs (default: 10)')
    parser.add_argument('--nb-steps', type=int, default=0, metavar='N',
                        help='number of steps (default: 0)')

    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))

    algo = import_module(params['algorithm']['module'])
    event_dir = os.path.join(rundir, 'events')
    node_desc = m.load(rundir, 'nodes.json')

    m.extend(rundir, 'simulator', {
      'nb-epochs': args.nb_epochs,
      'nb-steps': args.nb_steps
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
              'epoch': 0,
              'event-file': event_file,
              'train-set': [ train[i] for i in train_ind[rank] ],
              'train-iterator': None,
              'model': _model,
              'optimizer': optimizer
            })

            # Initialize topology
            topology = t.load(rundir)

        state, losses, done = algo.init(nodes, topology, params)
        
        # Initial logging
        if params["topology"]["name"] in ["fully-connected", "sample"]:
            log.state(nodes[0], state)
        else:
            for node in nodes:
                log.state(node, state)
        if params["logger"]["log-consensus-distance"]:
            log.log_consensus_distance(state)

        while True:  # Main loop
            state, losses, epoch_done, active_nodes = algo.next_step(state, params, rundir)
            log.loss(losses)

            if params["topology"]["name"] in ["fully-connected", "sample"] and should_log(nodes[0], epoch_done[0] if 0 in epoch_done else False, params, state):
                log.state(nodes[0], state)
            else:
                for active_node in active_nodes:
                    if should_log(active_node, epoch_done[active_node['rank']], params, state):
                        log.state(active_node, state)

            if all(epoch_done.values()) and params["logger"]["log-consensus-distance"]:
                log.log_consensus_distance(state)

            # Are we done?
            if args.nb_epochs and all([node['epoch'] >= args.nb_epochs for node in nodes]):
                break
            elif args.nb_steps and state['step'] == args.nb_steps:
                break

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
