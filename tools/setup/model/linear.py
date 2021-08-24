#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import torch
import torch.nn.functional as F
import setup.meta as m
import setup.dataset as ds


# Purely linear version of the convolution example of:
# https://github.com/seba-1511/dist_tuto.pth/
class Net(torch.nn.Module):
    """ Network architecture. """

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(input_size,10)
        self.input_size = input_size

    def forward(self, x, params):
        x = self.fc(x.view(-1, params['model']['input-size']))
        return F.log_softmax(x, dim=1)

    def copy(self):
        c = Net(self.input_size)
        for c1, s1 in zip(c.parameters(), self.parameters()):
            c1.mul_(0)
            c1.add_(s1)
        return c

def create(params):
    return Net(params['model']['input-size'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide Linear Model Options.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    dataset = m.params(rundir, 'dataset')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    m.extend(rundir, 'model', {
        'name': 'linear',
        'module': 'setup.model.linear',
        'input-size': ds.numbers[dataset['name']]['input-size']
    })

    if args.rundir is None:
        print(rundir)
