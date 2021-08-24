#!/usr/bin/env python
import torch
import argparse
import setup.meta as m
import logging
import setup.model as model
from torch.autograd import Variable
import torch.nn.functional as F

def average_models(models, weights=None):
    with torch.no_grad():
        if weights == None:
            weights = [ float(1./len(models)) for _ in range(len(models)) ]
        center_model = models[0].copy()
        for p in center_model.parameters():
            p.mul_(0)
        for m, w in zip(models, weights):
            for c1, p1 in zip(center_model.parameters(), m.parameters()):
                c1.add_(w*p1)
        return center_model

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
                for clique in topology.cliques:
                    models = [ nodes[rank]['model'] for rank in clique ]
                    clique_gradients = average_gradients(models)
                    update_gradients(models, clique_gradients)
                    for rank in clique:
                        nodes[rank]['optimizer'].step()
            elif params['algorithm']['unbiased-gradient']:
                for n in nodes:
                    rank = n['rank']
                    models = [ nodes[m]['model'] for m in args.averaging_neighbourhood[rank]]
                    gradients = average_gradients(models)
                    update_gradients([n['model']], gradients)
                    n['optimizer'].step()
            else:
                raise Exception("Incorrect call to 'd_psgd_unbiased_gradient_single_process', none of --clique-gradient or --unbiased-gradient  options provided")

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
            averaged[rank] = average_models(models, _weights) 

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
    state = { 'nodes': nodes, 'topology': topology, 'step': 0 }
    for n in nodes:
        n['train-iterator'] = iter(torch.utils.data.DataLoader(
            n['train-set'], 
            batch_size=int(params['algorithm']['batch-size']),
            shuffle=True
        ))
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
    
    # Apply Gradients
    gradient(nodes, topology, params)

    # Average with Neighbours
    average(nodes, topology, params)

    state['step'] += 1

    if all(epoch_done):
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
