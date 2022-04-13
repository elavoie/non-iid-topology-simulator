#!/usr/bin/env python
import copy
import itertools
import json
import os
from random import Random

import torch
import argparse
import setup.meta as m
import logging
import setup.model
import setup.topology as t
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


def average_gradients(models):
    with torch.no_grad():
        gradients = [ torch.zeros_like(p.grad.data) for p in models[0].parameters() ]
        for m in models:
            for g,p in zip(gradients, m.parameters()):
                g.add_(p.grad.data)
        for g in gradients:
            g.div_(len(models))
        return gradients

def update_models(models, new_model):
    with torch.no_grad():
        for m in models:
            for p, new_p in zip(m.parameters(), new_model.parameters()):
                p.mul_(0.)
                p.add_(new_p)
    return models

def update_gradients(models, gradients):
    with torch.no_grad():
        for m in models:
            for g,p in zip(gradients, m.parameters()):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                p.grad.data.zero_()
                p.grad.data.add_(g)
    return models

def gradient(nodes, topology, params):
    logging.info('  applying gradients') 
    if not params['algorithm']['clique-gradient'] and not params['algorithm']['unbiased-gradient']:
        logging.info('  applying own gradient') 
        for n in nodes:
            n['optimizer'].step()
    else:
        with torch.no_grad():
            if params['algorithm']['clique-gradient']:
                if not 'remove-clique-edges' in params['topology'].keys() \
                   or params['topology']['remove-clique-edges'] == 0:
                    for clique in topology['cliques']:
                        logging.info('  computing gradients for clique {}'.format(clique)) 
                        models = [ nodes[rank]['model'] for rank in clique ]
                        clique_gradients = average_gradients(models)
                        update_gradients(models, clique_gradients)
                        logging.info('  applying gradients') 
                        for rank in clique:
                            nodes[rank]['optimizer'].step()
                else:
                    edges = topology['edges']
                    for clique in topology['cliques']:
                        logging.info('  computing gradients for clique {} with less edges'.format(clique)) 
                        gradients = {}
                        for rank in clique:
                            # Average gradients only with other clique nodes
                            # with actual edges
                            models = [ nodes[r]['model'] for r in clique 
                                       if r == rank or r in edges[rank] ]
                            gradients[rank] = average_gradients(models)
                        logging.info('  applying gradients') 
                        for rank in clique:
                            update_gradients([nodes[rank]['model']], gradients[rank])
                            nodes[rank]['optimizer'].step()
            elif params['algorithm']['unbiased-gradient']:
                neighbourhoods = topology['neighbourhoods']
                gradients = {}
                for n in nodes:
                    rank = n['rank']
                    logging.info('  computing gradients for node {} with neighbourhood {}'.format(rank, neighbourhoods[rank])) 
                    models = [ nodes[m]['model'] for m in neighbourhoods[rank]]
                    gradients[rank] = average_gradients(models)
                for n in nodes:
                    logging.info('  applying gradient for node {}'.format(rank)) 
                    update_gradients([n['model']], gradients[n['rank']])
                    n['optimizer'].step()
            else:
                raise Exception('Invalid execution path, previous cases should cover all possibilities.')

def average(nodes, topology, params):
    logging.info('  computing averages of models')
    weights = topology['weights']
    edges = topology['edges']

    with torch.no_grad():
        averaged = [ None for _ in nodes ]

        # Compute averages
        for n in nodes:
            rank = n['rank']
            logging.info('  computing average for node {}'.format(rank))
            models = [ n['model'] ] + [ nodes[src]['model'] for src in edges[rank] ] 
            _weights = [ weights[rank,rank] ] + [ weights[src,rank] for src in edges[rank] ]
            averaged[rank] = setup.model.average(models, _weights) 

        # Update models
        for n in nodes:
            rank = n['rank']
            logging.info('  updating model for node {}'.format(rank))
            update_models([n['model']],averaged[rank])

def optimizer(model, params):
    return torch.optim.SGD(
            model.parameters(), 
            lr=params['algorithm']['learning-rate'], 
            momentum=params['algorithm']['learning-momentum'])

def init(nodes, topology, params):
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))
    logging.info('d-sgd.init')

    state = { 'nodes': nodes, 'topology': topology, 'step': 0 }
    for n in nodes:
        logging.info('d-sgd.init: creating data sampler for node {}'.format(n['rank']))
        n['train-iterator'] = iter(torch.utils.data.DataLoader(
            n['train-set'], 
            batch_size=int(params['algorithm']['batch-size']),
            shuffle=True
        ))

    if params['algorithm']['initial-averaging']:
        logging.info('d_sgd: averaging initial models')
        avg_model = setup.model.average([ nodes[rank]['model'] for rank in range(len(nodes)) ])
        for rank in range(len(nodes)):
            update_models([nodes[rank]['model']], avg_model)

    return (state, 0, False)


def it_has_next(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


cached_samples = {}


def get_sample(state, params, step):
    if step in cached_samples:
        return cached_samples[step]

    rnd = Random(42 + step)
    if params["topology"]["sample-method"] == "random":
        return rnd.sample(state['nodes'], params["topology"]["sample-size"])
    elif params["topology"]["sample-method"] == "random-with-overlap":
        if step == 0:  # The first sample is fully random
            return rnd.sample(state['nodes'], params["topology"]["sample-size"])
        else:
            # The sample at step x requires us to know the sample at step x-1 so we can derive overlap.
            active_nodes_last_step = get_sample(state, params, step - 1)
            active_nodes = rnd.sample(active_nodes_last_step, params["topology"]["sample-overlap"])
            eligible_nodes = [n for n in state['nodes'] if n not in active_nodes]
            active_nodes += rnd.sample(eligible_nodes, params["topology"]["sample-size"] - params["topology"]["sample-overlap"])

            cached_samples[step] = active_nodes
            return active_nodes


def next_step(state, params, rundir):
    logging.info('d-sgd.next step {}'.format(state['step']))
    active_nodes = state['nodes'] if params["topology"]["name"] != "sample" else get_sample(state, params, state["step"])
    topology = state['topology']

    # Local Training
    losses = {}
    epoch_done = {}
    for node in active_nodes:
        epoch_done_node = False
        logging.info('d-sgd.next computing gradient step on node {}'.format(node['rank']))
        data,target = node['train-iterator'].__next__()
        data, target = Variable(data), Variable(target)
        node['optimizer'].zero_grad()
        logging.info('d-sgd.next node {} forward propagation'.format(node['rank']))
        output = node['model'].forward(data, params)
        loss = F.nll_loss(output, target)
        logging.info('d-sgd.next node {} backward propagation'.format(node['rank']))
        loss.backward()
        losses[node['rank']] = loss.tolist()

        # Reset the iterator if needed
        res = it_has_next(node['train-iterator'])
        if res is None:
            # Epoch complete - reset the iterator
            epoch_done_node = True
            node['epoch'] += 1
            node['train-iterator'] = iter(torch.utils.data.DataLoader(
                node['train-set'],
                batch_size=int(params['algorithm']['batch-size']),
                shuffle=True
            ))
        else:
            node['train-iterator'] = res

        epoch_done[node['rank']] = epoch_done_node

    if params["topology"]["name"] != "sample":
        # Apply Gradients
        gradient(active_nodes, topology, params)

        # Average with Neighbours
        average(active_nodes, topology, params)

        # Randomize the topology if needed
        if params["topology"]["name"] == "random-graph" and params["topology"]["randomize"]:
            logging.info("Randomizing neighbours")
            params['topology']['topology-seed'] += 1  # To make sure that a new graph is generated

            from setup.topology.random_graph import generate_topology
            new_topology = generate_topology(state['nodes'], params)
            with open(os.path.join(rundir, 'topology.json'), 'w+') as topology_file:
                json.dump(new_topology, topology_file)
            edges = new_topology['edges']
            new_topology['edges'] = {int(rank): edges[rank] for rank in edges}
            new_topology['weights'] = torch.tensor(new_topology['weights'])
            state['topology'] = new_topology
    else:
        # Apply gradients
        for n in active_nodes:
            n['optimizer'].step()

        # Average the model and replace the model of other nodes with the aggregated one
        with torch.no_grad():
            # Compute averages of the models of active nodes
            logging.info(f'  computing average model of sample consisting of {len(active_nodes)} nodes')
            models = [n['model'] for n in active_nodes]
            weights = [1 / len(active_nodes) for _ in active_nodes]
            avg_model = setup.model.average(models, weights)

            # Replace the current model of every node with this one
            all_models = [n['model'] for n in state['nodes']]
            update_models(all_models, avg_model)

    state['step'] += 1

    return state, losses, epoch_done, active_nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide Options for D-SGD Optimization Algorithm.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run in which to save the dataset options.')
    parser.add_argument('--learning-rate', type=float, default=0.1,
            help='Magnitude of gradient step (default: 0.1).')
    parser.add_argument('--learning-momentum', type=float, default=0.0,
            help='Magnitude of momentum (default: 0.0).')
    parser.add_argument('--batch-size', type=int, default=128,
            help='Maximum number of samles to use for every gradient step (default: 128).')
    parser.add_argument('--initial-averaging', action='store_const', const=True, default=False, 
            help="Average all models before training ( default: False)")
    parser.add_argument('--clique-gradient', action='store_const', const=True, default=False, 
            help="Use the average gradient of the clique, instead of the local one. Only works with one of the clique topologies (ex: clique-ring, fully-connected-cliques, fractal-cliques ) ( default: False)")
    parser.add_argument('--unbiased-gradient', action='store_const', const=True, default=False, 
            help="Use the average gradient of a subset of neighbours representing equally all classes. Only works with the 'greedy-diverse-10' topology ( default: False)")
    args = parser.parse_args()
    rundir = m.rundir(args)
    params = m.params(rundir)
    topology = t.load(rundir)

    if args.clique_gradient:
        assert 'cliques' in topology.keys(), \
           "Invalid --clique-gradient with {} topology".format(params['topology']['name']) +\
           ", no 'cliques' found in topology.json."

    if args.unbiased_gradient:
        assert 'neighbourhoods' in topology.keys(), \
           "Invalid --unbiased-gradient with {} topology".format(params['topology']['name']) +\
           ", no 'neighbourhoods' found in topology.json."

    algorithm = {
        'name': 'd-sgd',
        'module': 'simulate.algorithm.d_sgd',
        'learning-rate': args.learning_rate,
        'learning-momentum': args.learning_momentum,
        'batch-size': args.batch_size,
        'initial-averaging': args.initial_averaging,
        'clique-gradient': args.clique_gradient,
        'unbiased-gradient': args.unbiased_gradient,
    }
    m.extend(rundir, 'algorithm', algorithm) # Add to run parameters

    if args.rundir is None:
        print(rundir)
