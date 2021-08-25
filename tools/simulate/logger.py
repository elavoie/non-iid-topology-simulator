#!/usr/bin/env python
import argparse
import logging
import json
import pickle
import copy
from importlib import import_module
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.multiprocessing import Process, JoinableQueue, Pipe
import setup.meta as m
import setup.dataset as d

def model_accuracy(model, dataset, params):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset:
            output = model.forward(data, params)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataset)
    return float(correct)/float(len(dataset)), test_loss

def log_task(tasks, params):
    valid_set = torch.utils.data.DataLoader(d.valid(params), 100)
    test_set = torch.utils.data.DataLoader(d.test(params), 100)
    model = import_module(params['model']['module']).create(params)
    for rank, epoch, step, model_state, event_file in iter(tasks.get, 'STOP'):
        model.load_state_dict(pickle.loads(model_state))

        for name, dataset in [('valid', valid_set), ('test', test_set)]:
            accuracy, test_loss = model_accuracy(model,dataset,params)
            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": name,
                    "rank": rank,
                    "epoch": epoch,
                    "batch": step,
                    "loss": test_loss,
                    "accuracy": accuracy 
                }) + '\n')
        tasks.task_done()

class Logger:
    def __init__(self, params, nodes):
        nb_nodes = len(nodes)
        self.params = params
        self.running_loss = [ 0.0 for _ in range(nb_nodes) ]
        self.running_loss_count = 0

        # Create empty files first to avoid race condition on first write
        for n in nodes:
            open(n['event-file'], 'a').close()

        self.tasks = JoinableQueue(maxsize=nb_nodes)
        self.processes = []
        for _ in range(params['logger']['nb-processes']):
            p = Process(target=log_task, args=(self.tasks, copy.deepcopy(params)))
            p.start()
            self.processes.append(p)

    def state(self, epoch, state):
        nodes = state['nodes']
        if epoch > 0:
            self.log_train_accuracy(epoch, state)
        self.log_test_accuracy(epoch, state) 

    def loss(self, loss):
        assert len(loss) == len(self.running_loss), \
            "Inconsistent loss vector, expected {} values but got {}".format(
            len(self.running_loss), len(loss))
        for i in range(len(loss)):
            self.running_loss[i] += loss[i]
        self.running_loss_count += 1

    def log_train_accuracy(self, epoch, state):
        nodes = state['nodes']
        params = self.params
        for n in nodes:
            rank = n['rank']
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

            if self.running_loss_count > 0:
                running_loss = self.running_loss[rank] / self.running_loss_count
                self.running_loss[rank] = 0.0
            else:
                running_loss = 0.0

            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": "train",
                    "rank": rank,
                    "epoch": epoch,
                    "batch": state['step'],
                    "loss": total_loss/num_batches,
                    "running_loss": running_loss,
                    "accuracy": correct / example_number 
                }) + '\n')

        self.running_loss_count = 0

    def log_test_accuracy(self, epoch, state):
        nodes = state['nodes']
        params = self.params

        for n in nodes:
            rank = n['rank']
            model = n['model']
            event_file = n['event-file']
            step = state['step']
            self.tasks.put((n['rank'], epoch, step, pickle.dumps(n['model'].state_dict()), event_file))
        self.tasks.join()
        
    def stop(self):
        for _ in range(self.params['logger']['nb-processes']): 
            self.tasks.put('STOP')
        for p in self.processes:
            p.join()

def init(params, nodes):
    return Logger(params, nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log events happening during simulation.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run in which to save options.')
    parser.add_argument('--nb-processes', type=int, default=4, metavar='N',
            help='Number of parallel processes to log the accuracy of models. (default: 8)')

    args = parser.parse_args()
    rundir = m.rundir(args)

    logger = {
        'nb-processes': args.nb_processes
    }
    m.extend(rundir, 'logger', logger) # Add to run parameters

    if args.rundir is None:
        print(rundir)
