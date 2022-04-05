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
from math import floor

class GN_LeNet(nn.Module):
    """
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    Layer parameters taken from: https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/1parts/gnlenet_train_val.prototxt.template
    (with Group Normalisation).
    Results for previous model described in http://proceedings.mlr.press/v119/hsieh20a.html
    """

    def __init__(self, params, input_channel=3, output=10, model_input=(24,24)):
        super(GN_LeNet, self).__init__()

        self.params = params
        self.input_channel = input_channel
        self.output = output
        self.model_input = model_input
        self.classifier_input = classifier_input_calculator(*model_input)

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
            nn.Linear(self.classifier_input, output),
        )

    def forward(self, x, params):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def classifier_input_calculator(y,z):
    """Given the input shape of GN_Lenet, returns the size of the output of the Sequential module.
    This function is helpful to compute the size of the input of the classifier.
    Args:
        - y : the size of the first dimension of a channel in an input data
        - z : the size of the second dimension of a channel in an input data
    Output:
        - The size of the output of the sequential module of the GN_Lenet
    
    Example:
        Given an image from MNIST, since the images have shape (1,24,24), the size of a single
        channel is (24,24). classifier_input_calculator(24,24) == 256 and 256 is indeed the
        required input for the classifier.
    """

    def down(x,y,z):
        """Computes the final shape of a 3D tensor of shape (x,y,z) after the Conv2d and
        Maxpool layers in the gn_lenet model.

        Args:
            x (Int): 1st dimension of the input tensor
            y (Int): 2nd dimension
            z (Int): 3rd dimension

        Returns:
            (Int, Int, Int): Shape of the output of the Conv2d + Maxpool Layer.
        """
        return x,floor((y-3)/2)+1,floor((z-3)/2)+1

    x,y,z = down(32,y,z)
    x,y,z = down(x,y,z)
    x,y,z = down(2*x,y,z)

    return x*y*z


def create(params):
    if params['dataset']['name'] == 'mnist':
        input_channel = 1
        output = 10
        model_input = (24,24)
    elif params['dataset']['name'] == 'cifar10':
        input_channel = 3
        output = 10
        model_input = (32,32)
    elif params['dataset']['name'] == 'svhn':
        input_channel = 3
        output = 10
        model_input = (32,32)
    else:
        raise Exception("Invalid dataset: {}".format(args.dataset))
    return GN_LeNet(params, input_channel, output, model_input)

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

