#!/usr/bin/env python
import argparse
import logging
import json
import pickle
import copy
import os
import math
import time
from importlib import import_module
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.multiprocessing import Process, JoinableQueue, Pipe
import setup.meta as m
import setup.dataset as d
import setup.model
import statistics

def model_accuracy(model, dataset, params):
    model.eval()
    correct = 0
    example_number = 0
    num_batches = 0
    total_loss = 0.0

    if len(dataset) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for data, target in dataset:
            output = model.forward(data, params)
            total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            example_number += target.size(0)
            num_batches += 1
    return float(correct)/float(example_number), total_loss / float(example_number)

def model_distance(model1, model2):
    # Compute Euclidean distance (L2-norm) considering all parameters
    with torch.no_grad():
        ds = torch.tensor([
            torch.sum((p1 - p2)**2) for p1, p2 in zip(model1.parameters(), model2.parameters())
        ])
        return torch.sqrt(torch.sum(ds)).tolist()

def log_task(tasks, params):
    valid_set = torch.utils.data.DataLoader(d.valid(params), 100)
    test_set = torch.utils.data.DataLoader(d.test(params), 100)
    model = import_module(params['model']['module']).create(params)
    for rank, epoch, step, model_state, event_file in iter(tasks.get, 'STOP'):
        model.load_state_dict(pickle.loads(model_state))
        logging.info('logger.log_task logging model accuracy for node {}'.format(rank))

        sets = []
        if not params['logger']['skip-validation']:
            sets.append(('valid', valid_set))

        if not params['logger']['skip-testing']:
            sets.append(('test', test_set))

        for name, dataset in sets:
            accuracy, test_loss = model_accuracy(model,dataset,params)
            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": name,
                    "rank": rank,
                    "epoch": epoch,
                    "step": step,
                    "loss": test_loss,
                    "accuracy": accuracy,
                    "timestamp": m.now()
                }) + '\n')
        tasks.task_done()

class Logger:
    def __init__(self, params, rundir):
        nb_nodes = params['nodes']['nb-nodes']
        self.params = params
        self.rundir = rundir
        self.running_loss = [ 0.0 for _ in range(nb_nodes) ]
        self.running_loss_count = [0 for _ in range(nb_nodes)]
        self.global_events = os.path.join(rundir, 'events', 'global.jsonlines')

        self.tasks = JoinableQueue(maxsize=nb_nodes)
        self.processes = []
        for i in range(params['logger']['nb-processes']):
            logging.info('starting logging task {}'.format(i))
            p = Process(target=log_task, args=(self.tasks, copy.deepcopy(params)))
            p.start()
            self.processes.append(p)

    def state(self, epoch, state):
        if epoch > 0:
            self.log_train_accuracy(state)
        if epoch % self.params['logger']['accuracy-logging-interval'] == 0:
            self.log_test_accuracy(state)
        self.log_consensus_distance(state)

    def loss(self, losses):
        for node_rank, loss in losses.items():
            self.running_loss[node_rank] += loss
            self.running_loss_count[node_rank] += 1

    def log_train_accuracy(self, state):
        if self.params['logger']['skip-training']:
            return

        nodes = state['nodes']
        params = self.params
        for n in nodes:
            rank = n['rank']
            epoch = n['epoch']
            logging.info('logger.log_train_accuracy node {}'.format(rank))
            model = n['model']
            model.eval()
            event_file = n['event-file']
            total_loss = 0.0
            num_batches = 0.0
            correct = 0.0
            example_number = 0.0
            train = torch.utils.data.DataLoader(n['train-set'], 1000)

            with torch.no_grad():
                for data, target in train:
                    data, target = Variable(data), Variable(target)
                    output = model.forward(data, params)
                    loss = F.nll_loss(output, target)
                    total_loss += loss.item()
                    num_batches += 1.0
                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == target).sum().item()
                    example_number += target.size(0)

            if self.running_loss_count[rank] > 0:
                running_loss = self.running_loss[rank] / self.running_loss_count[rank]
                self.running_loss[rank] = 0.0
            else:
                running_loss = 0.0

            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": "train",
                    "rank": rank,
                    "epoch": epoch,
                    "step": state['step'],
                    "loss": total_loss/num_batches,
                    "running_loss": running_loss,
                    "accuracy": correct / example_number,
                    "timestamp": m.now()
                }) + '\n')

            self.running_loss_count[rank] = 0

    def log_test_accuracy(self, state):
        nodes = state['nodes']
        params = self.params

        # If we have a fully connected topology, it is wasteful to compute the test accuracy for all nodes since they
        # are all the same. So we only compute it for one node
        if params["topology"]["name"] == "fully-connected":
            n = nodes[0]
            self.tasks.put((n['rank'], n['epoch'], state['step'], pickle.dumps(n['model'].state_dict()), n['event-file']))
        else:
            for n in nodes:
                event_file = n['event-file']
                step = state['step']
                self.tasks.put((n['rank'], n['epoch'], step, pickle.dumps(n['model'].state_dict()), event_file))
        self.tasks.join()
        
    def log_consensus_distance(self, state):
        logging.info('logger.log_consensus_distance')
        models = [ n['model'] for n in state['nodes'] ]
        center = setup.model.average(models)
        distances = [ model_distance(center, m) for m in models ]
        avg = statistics.mean(distances)
        std = statistics.stdev(distances) if len(distances) > 1 else 0.

        with torch.no_grad():
            norm = math.sqrt(sum([ torch.sum(p**2).tolist() for p in center.parameters() ]))

        with open(self.global_events, 'a') as events:
            events.write(json.dumps({
                "type": "consensus-distance",
                "step": state['step'],
                "distance_to_center": {
                    "global": {
                        "avg": avg,
                        "std": std,
                        "max": max(distances),
                        "min": min(distances)
                    }
                },
                "center": {
                    "norm": norm
                },
                "timestamp": m.now()
            }) + '\n')
        
    def stop(self):
        for _ in range(self.params['logger']['nb-processes']): 
            self.tasks.put('STOP')
        for p in self.processes:
            p.join()

def init(params, rundir):
    logging.basicConfig(level=getattr(logging, params['meta']['log'].upper(), None))
    return Logger(params, rundir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log events happening during simulation.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run in which to save options.')
    parser.add_argument('--nb-processes', type=int, default=4, metavar='N',
            help='Number of parallel processes to log the accuracy of models. (default: 8)')
    parser.add_argument('--accuracy-logging-interval', type=int, default=1, metavar='N',
                        help='Log validation and test accuracy every X epochs. (default: 1)')
    parser.add_argument('--skip-validation', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on validation set. ( default: False)")
    parser.add_argument('--skip-testing', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on test set. ( default: False)")
    parser.add_argument('--skip-training', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on training set. ( default: False)")

    args = parser.parse_args()
    rundir = m.rundir(args)

    logger = {
        'nb-processes': args.nb_processes,
        'accuracy-logging-interval': args.accuracy_logging_interval,
        'skip-testing': args.skip_testing,
        'skip-validation': args.skip_validation,
        'skip-training': args.skip_training,
    }
    m.extend(rundir, 'logger', logger) # Add to run parameters

    if args.rundir is None:
        print(rundir)
