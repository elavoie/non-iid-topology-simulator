#!/usr/bin/env python
import os
import sys
import argparse
import logging
import functools
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import setup.meta as m
import setup.dataset as ds

class GN_LeNet(nn.Module):
    """
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    Layer parameters taken from: https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/1parts/gnlenet_train_val.prototxt.template
    (with Group Normalisation).
    Results for previous model described in http://proceedings.mlr.press/v119/hsieh20a.html
    """

    def __init__(self, params, input_channel=3, output=10, classifier_input=576):
        super(GN_LeNet, self).__init__()

        self.params = params
        self.input_channel = input_channel
        self.output = output
        self.classifier_input = classifier_input

        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, output),
        )

    def forward(self, x, params):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def copy(self):
        c = GN_LeNet(self.params, self.input_channel, self.output, self.classifier_input)
        for c1, s1 in zip(c.parameters(), self.parameters()):
            c1.mul_(0)
            c1.add_(s1)
        return c

def create(params):
    if params['dataset']['name'] == 'mnist':
        input_channel = 1
        output = 10
        classifier_input = 256
    elif params['dataset']['name'] == 'cifar10':
        input_channel = 3
        output = 10
        classifier_input = 576
    else:
        raise Exception("Invalid dataset: {}".format(args.dataset))
    return GN_LeNet(params, input_channel, output, classifier_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide Group-Normalized LeNet Model Options.')
    parser.add_argument('--rundir', type=str, default=None,
      help='Directory of the run in which to save the topology options.')
    args = parser.parse_args()
    rundir = m.rundir(args)
    meta = m.params(rundir, 'meta')
    dataset = m.params(rundir, 'dataset')
    logging.basicConfig(level=getattr(logging, meta['log'].upper(), None))

    m.extend(rundir, 'model', {
        'name': 'gn-lenet',
        'module': 'setup.model.gn_lenet'
    })

    if args.rundir is None:
        print(rundir)
