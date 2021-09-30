#!/usr/bin/env python
import torch
import argparse
import setup.meta as m
import logging
import setup.model
from torch.autograd import Variable
import torch.nn.functional as F

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
    if not params['algorithm']['clique-gradient'] and not params['algorithm']['unbiased-gradient']:
        for n in nodes:
            n['optimizer'].step()
    else:
        with torch.no_grad():
            if params['algorithm']['clique-gradient']:
                if not 'remove-clique-edges' in params['topology'].keys() \
                   or params['topology']['remove-clique-edges'] == 0:
                    for clique in topology['cliques']:
                        models = [ nodes[rank]['model'] for rank in clique ]
                        clique_gradients = average_gradients(models)
                        update_gradients(models, clique_gradients)
                        for rank in clique:
                            nodes[rank]['optimizer'].step()
                else:
                    edges = topology['edges']
                    for clique in topology['cliques']:
                        gradients = {}
                        for rank in clique:
                            # Average gradients only with other clique nodes
                            # with actual edges
                            models = [ nodes[r]['model'] for r in clique 
                                       if r == rank or r in edges[rank] ]
                            gradients[rank] = average_gradients(models)
                        for rank in clique:
                            update_gradients([nodes[rank]['model']], gradients[rank])
                            nodes[rank]['optimizer'].step()
            elif params['algorithm']['unbiased-gradient']:
                neighbourhoods = topology['neighbourhoods']
                for n in nodes:
                    rank = n['rank']
                    models = [ nodes[m]['model'] for m in neighbourhoods[rank]]
                    gradients = average_gradients(models)
                    update_gradients([n['model']], gradients)
                    n['optimizer'].step()
            else:
                raise Exception('Invalid execution path, previous cases should cover all possibilities.')

def average(nodes, topology, params):
    weights = topology['weights']
    edges = topology['edges']

    with torch.no_grad():
        averaged = [ None for _ in nodes ]

        # Compute averages
        for n in nodes:
            rank = n['rank']
            models = [ n['model'] ] + [ nodes[src]['model'] for src in edges[rank] ] 
            _weights = [ weights[rank,rank] ] + [ weights[src,rank] for src in edges[rank] ]
            averaged[rank] = setup.model.average(models, _weights) 

        # Update models
        for n in nodes:
            rank = n['rank']
            update_models([n['model']],averaged[rank])

def optimizer(model, params):
    return torch.optim.SGD(
            model.parameters(), 
            lr=params['algorithm']['learning-rate'], 
            momentum=params['algorithm']['learning-momentum'])

def init(nodes, topology, params):
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))
    state = { 'nodes': nodes, 'topology': topology, 'step': 0 }
    for n in nodes:
        n['train-iterator'] = iter(torch.utils.data.DataLoader(
            n['train-set'], 
            batch_size=int(params['algorithm']['batch-size']),
            shuffle=True
        ))

    if params['algorithm']['initial-averaging']:
        logging.info('d_sgd: averaging initial models')
        avg_model = average_models([ nodes[rank]['model'] for rank in range(len(nodes)) ])
        for rank in range(len(nodes)):
            update_models([nodes[rank]['model']], avg_model)

    return (state, 0, False)

def next(state, params):
    nodes = state['nodes']
    topology = state['topology']

    # Local Training
    losses = []
    epoch_done = []
    for node in nodes:
        try:
            data,target = node['train-iterator'].__next__()
            data, target = Variable(data), Variable(target)
            node['optimizer'].zero_grad()
            output = node['model'].forward(data, params)
            loss = F.nll_loss(output, target)
            loss.backward()
            losses.append(loss.tolist())
            epoch_done.append(False)
        except StopIteration:
            epoch_done.append(True)

    assert all(epoch_done) or not any(epoch_done), "Some nodes completed their epoch before others."
   
    if not all(epoch_done):
        # Apply Gradients
        gradient(nodes, topology, params)

        # Average with Neighbours
        average(nodes, topology, params)

        state['step'] += 1
    else:
        for n in nodes:
            n['train-iterator'] = iter(torch.utils.data.DataLoader(
                n['train-set'], 
                batch_size=int(params['algorithm']['batch-size']),
                shuffle=True
            ))

    return (state, losses, all(epoch_done))

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

    if args.clique_gradient:
        assert 'cliques' in params['topology'].keys(), \
           "Invalid --clique-gradient with {} topology".format(params['topology']['name']) +\
           ", no 'cliques' found in topology.json."

    if args.unbiased_gradient:
        assert 'neighbourhoods' in params['topology'].keys(), \
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
