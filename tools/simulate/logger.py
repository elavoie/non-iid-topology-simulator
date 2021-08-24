#!/usr/bin/env python
import argparse
import logging
import json
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import setup.meta as m
import setup.dataset as d

class Logger:
    def __init__(self, params):
        self.params = params
        self.running_loss = [ 0.0 for _ in range(params['nodes']['nb-nodes']) ]
        self.running_loss_count = 0
        self.valid_set = torch.utils.data.DataLoader(d.valid(params), 1000)
        self.test_set = torch.utils.data.DataLoader(d.test(params), 1000)

    def state(self, epoch, state):
        nodes = state['nodes']
        if epoch > 0:
            self.log_train_accuracy(epoch, state)
        self.log_test_accuracy(epoch, state, name='test') 
        self.log_test_accuracy(epoch, state, name='valid') 

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

    def log_test_accuracy(self, epoch, state, name='test'):
        nodes = state['nodes']
        params = self.params
        assert name == 'test' or name == 'valid', 'Invalid test dataset {}'.format(name)
        test_set = self.test_set if name == 'test' else self.valid_set

        for n in nodes:
            rank = n['rank']
            model = n['model']
            model.eval()
            event_file = n['event-file']
            test_loss = 0
            correct = 0

            with torch.no_grad():
                for data, target in test_set:
                    output = model.forward(data, params)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_set.dataset)

            if name != 'test':
                logging.info('(Rank {}, Epoch {}, Batch {}) {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    rank, epoch, state['step'], name, test_loss, correct, len(test_set.dataset),
                    100. * correct / len(test_set.dataset)))
            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": name,
                    "rank": rank,
                    "epoch": epoch,
                    "batch": state['step'],
                    "loss": test_loss,
                    "accuracy": float(correct)/float(len(test_set.dataset))
                }) + '\n')

def init(params):
    return Logger(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log events happening during simulation.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run in which to save options.')

    args = parser.parse_args()
    rundir = m.rundir(args)

    logger = {
    }
    m.extend(rundir, 'logger', logger) # Add to run parameters

    if args.rundir is None:
        print(rundir)
