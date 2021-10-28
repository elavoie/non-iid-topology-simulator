#!/usr/bin/env python
import math
import os
import sys
import argparse
import setup.meta as m
import setup.nodes as nodes
import logging
from random import Random
import torch
from torchvision import datasets, transforms

numbers = {
  'mnist': {},
  'cifar10': {},
  'svhn':{}
}

numbers['mnist']['input-size'] = 784
numbers['mnist']['classes'] = [ i for i in range(10) ]

# For future reference, the number of examples of each class in the original
# MNIST training set are:
numbers['mnist']['original_train_set_number_of_examples'] = [
    5923,
    6742,
    5958,
    6131,
    5842,
    5421,
    5918,
    6265,
    5851,
    5949
]

# The number of examples of each class in the original MNIST test set. 
# The same number are used for the validation set:
numbers['mnist']['original_test_set_number_of_examples'] = [
    980,
    1135,
    1032,
    1010,
    982,
    892,
    958,
    1028,
    974,
    1009
 ]
numbers['mnist']['val_set_number_of_examples'] = numbers['mnist']['original_test_set_number_of_examples']

# The number of examples of each class in the final training set (taken from
# the original training set):
numbers['mnist']['train_set_number_of_examples'] =  [
    4943,
    5607,
    4926,
    5121,
    4860,
    4529,
    4960,
    5237,
    4877,
    4940
]

numbers['cifar10']['input-size'] = 3072
numbers['cifar10']['classes'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
numbers['cifar10']['original_train_set_number_of_examples'] = [ 5000 for _ in range(10) ] 
numbers['cifar10']['original_test_set_number_of_examples'] = [ 1000 for _ in range(10) ]
numbers['cifar10']['val_set_number_of_examples'] = numbers['cifar10']['original_test_set_number_of_examples']
numbers['cifar10']['train_set_number_of_examples'] = [ 4000 for _ in range(10) ]

numbers['svhn']['input-size'] = 32*32*3
numbers['svhn']['classes'] = [ i for i in range(10) ]
numbers['svhn']['original_train_set_number_of_examples'] = [ 5000 for _ in range(10) ] 
numbers['svhn']['original_test_set_number_of_examples'] = [ 1000 for _ in range(10) ]
numbers['svhn']['val_set_number_of_examples'] = numbers['svhn']['original_test_set_number_of_examples']
numbers['svhn']['train_set_number_of_examples'] = [ 4000 for _ in range(10) ]

def validate(dataset_params):
    dataset_name = dataset_params['name']
    assert dataset_params['name'] == 'mnist'  or dataset_params['name'] == 'cifar10' \
         or dataset_params['name'] == 'svhn',\
        'Unsupported dataset_params {}'.format(dataset_params['name'])
    assert len(dataset_params['train-examples-per-class']) == len(numbers[dataset_name]['classes']),\
                "Expected {} train-examples-per-class numbers, got {} instead.".format(\
                  len(numbers[dataset_name]['classes']),\
                  len(dataset_params['train-examples-per-class']))
    assert len(dataset_params['validation-examples-per-class']) == len(numbers[dataset_name]['classes']),\
            "Expected {} validation-examples-per-class numbers, got {} instead.".format(\
              len(numbers[dataset_name]['classes']),\
              len(dataset_params['validation-examples-per-class']))
    return dataset_params

def download(dataset_params):
    if dataset_params['name'] == 'mnist':
        data = datasets.MNIST(
            dataset_params['data-directory'],
            download=True)
    elif dataset_params['name'] == 'cifar10':
        data = datasets.CIFAR10(
            dataset_params['data-directory'],
            download=True)
    elif dataset_params['name'] == 'svhn':
        data = datasets.SVHN(
            dataset_params['data-directory'],
            download=True)
    else:
        print('dataset.download: Invalid dataset {}'.format(dataset_params['name']))
        sys.exit(1)
    return data

def params(_params):
    if type(_params) == str:
        params = m.params(_params)
    elif type(_params) == dict:
        assert 'dataset' in _params.keys(), "Invalid _params dictionary, should have " +\
        "a 'dataset' property"
        params = _params['dataset']
    else:
        raise Exception('Invalid _params parameter, should be path to a run directory or a ' +\
              'dictionary instead of {}'.format(_params))
    return validate(params)

def train(_params): 
    dataset = params(_params)
    if dataset['name'] == 'mnist':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        train = datasets.MNIST(
            dataset['data-directory'],
            train=True,
            download=True,
            transform=transform)
    elif dataset['name'] == 'cifar10':
        # Transforms taken from this tutorial: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
        transform = transforms.Compose([
            #transforms.Pad(4),
            #RandomHorizontalFlip(random=rand), # Reimplemented to be able to use a deterministic seed
            #RandomCrop(32, random=rand),       # Reimplemented to be able to use a deterministic seed
            transforms.ToTensor()])
        train = datasets.CIFAR10(
            dataset['data-directory'],
            train=True,
            download=True,
            transform=transform)
        logging.info(train)
        logging.info(len(train.targets))
        logging.info(type(train.targets))
    elif dataset['name'] == 'svhn':
        transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalization values taken from this project: https://github.com/ucbdrive/skipnet/blob/master/cifar/data.py
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        train = datasets.SVHN(
            dataset['data-directory'],
            split="train",
            download=True,
            transform=transform)
        train.targets = train.labels
    else:
        print('Unsupported dataset {}'.format(dataset['name']))
        sys.exit(1)
    return train

def valid(_params):
    _, valid_ind = partition([], _params) 
    train_set = train(_params)
    return [ train_set[i] for i in valid_ind ]

def test(_params):
    dataset = params(_params)
    if dataset['name'] == 'svhn':
        transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalization values taken from this project: https://github.com/ucbdrive/skipnet/blob/master/cifar/data.py
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        test = datasets.SVHN(
            dataset['data-directory'],
            split="test",
            download=True,
            transform=transform)
        test.targets = test.labels
    elif dataset['name'] == 'mnist':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        test = datasets.MNIST(
            dataset['data-directory'],
            train=False,
            download=True,
            transform=transform)
    elif dataset['name'] == 'cifar10':
        # Transforms taken from this tutorial: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
        transform = transforms.Compose([
            #transforms.Pad(4),
            #RandomHorizontalFlip(random=rand), # Reimplemented to be able to use a deterministic seed
            #RandomCrop(32, random=rand),       # Reimplemented to be able to use a deterministic seed
            transforms.ToTensor()])
        test = datasets.CIFAR10(
            dataset['data-directory'],
            train=False,
            download=True,
            transform=transform)
    else:
        print('Unsupported dataset {}'.format(dataset['name']))
        sys.exit(1)
    return test

def log_train_indexes(rank, train_indexes):
    print("Saving train samples in ./debug/train/{}.samples, 'cat debug/train/*\.samples | sort -n | uniq | wc -l' should give 50000 if you intend to have all training examples present.".format(rank))
    debug_file = os.path.join(os.getcwd(), "debug", "train", "{}.samples".format(rank))
    with open(debug_file, 'w') as debug:
        debug.write('\n'.join(map(str, train_indexes)) + '\n')

def log_validation_indexes(val_indexes):
    print("Saving validation samples in ./debug/valid.samples, 'cat debug/valid.samples | sort -n | uniq | wc -l' should give 10000.")
    debug_file = os.path.join(os.getcwd(), "debug", "valid.samples")
    with open(debug_file, 'w') as debug:
        debug.write('\n'.join(map(str, val_indexes)) + '\n')

# Partition the training dataset into local datasets for each node,
# and a validation set common to all nodes (e.g. for hyper-parameter tuning),
# using ranges generated by setup.nodes.assign_ranges.
#
# Input: 
#       1. Node ranges (each node has a list of index ranges (start, end), one per class)
#       2. Run params (requires dataset and nodes params)
# Output: 
#       1. List of indexes of train examples for each node.
#       2. List of indexes of train examples used for validation.
def partition(node_ranges, _params):
    # Retrieve parameters
    _train = train(_params) 
    dataset_params = params(_params)
    nb_classes = dataset_params['nb-classes']
    dataset_name = dataset_params['name']
    validation_examples_per_class = dataset_params['validation-examples-per-class'] 
    train_set_number_of_examples = numbers[dataset_name]['original_train_set_number_of_examples']
    nodes_params = nodes.params(_params)
    seed = _params['meta']['seed']
    total_of_examples = nodes_params['total-of-examples']
    
    # Deterministically Initialize Pseudo-Random Number Generator
    rand = Random() 
    rand.seed(seed)

    logging.info('partition: split the dataset per class')
    indexes = { x: [] for x in range(nb_classes) }
    if type(_train.targets) != torch.Tensor:
        targets = torch.tensor(_train.targets)
    else:
        targets = _train.targets
    for x in indexes:
        c = (targets.clone().detach() == x).nonzero()
        indexes[x] = c.view(len(c)).tolist()

    # Store the classes for every index for sanity checking later
    classes = { i: targets[i].tolist() \
                for i in range(len(targets)) }

    # We create a validation set from the training examples
    # to tune hyper-parameters. The validation set is the same
    # for all experiments (similar to the test set).
    logging.info('partition: choosing validation examples')
    rand_val = Random()
    rand_val.seed(1337) 
    val_indexes = [] 
    for c in range(nb_classes):
        rand_val.shuffle(indexes[c])
        upper_val_index = validation_examples_per_class[c]
        val_indexes.extend(indexes[c][0:upper_val_index])
        indexes[c] = indexes[c][upper_val_index:]
        if all(map(lambda x: x == 0, validation_examples_per_class)):
            assert len(indexes[c]) == train_set_number_of_examples[c],\
                "Expected train set for class {} to have {} examples instead of {}".format(
                    c, 
                    train_set_number_of_examples[c], 
                    len(indexes[c])
                )
    logging.info('partition: validation set size {}'.format(len(val_indexes)))
    
    # We shuffle the list of indexes for each class so that a range of indexes
    # from the shuffled list corresponds to a random sample (without
    # replacement) from the list of examples.  This makes sampling faster in
    # the next step. 
    #
    # Additionally, we append additional and different shufflings of the same
    # list of examples to cover the total number of examples assigned
    # when it is larger than the number of available examples.
    logging.info('partition: shuffling examples')
    shuffled = []
    for c in range(nb_classes):
        ind_len = len(indexes[c])
        min_len = max(ind_len,total_of_examples[c])
        shuffled_c = []
        for i in range(int(math.ceil(min_len/ind_len))):
            shuffled_c.extend(rand.sample(indexes[c], ind_len))
        shuffled.append(shuffled_c)

    # Sampling examples for each node now simply corresponds to extracting
    # the assigned range of examples for that node.
    logging.info('partition: sampling examples for each node')
    partition = []
    for ranges in node_ranges:
        local = []
        for c in range(nb_classes):
            start,end = tuple(ranges[c])
            local.extend(shuffled[c][start:end])

        if dataset_params['log-partition-indexes']:
            log_train_indexes(len(partition), local)
        partition.append(local)

    if dataset_params['log-partition-indexes']:
        log_validation_indexes(val_indexes)

    # Sanity checking of class frequency for each node
    for n in range(len(partition)):
        local = partition[n]
        freq = { c: 0 for c in range(nb_classes) }
        for i in local:
            freq[classes[i]] += 1

        ranges = node_ranges[n]
        for c in range(nb_classes):
            start,end = tuple(ranges[c])
            assert freq[c] == end-start, 'Error in partitioning'

    # Sanity checking that we are using distinct examples up to the
    # number of available examples
    if len(node_ranges) > 0:
        all_indexes = set()
        for local in partition:
            all_indexes.update(local)
        max_indexes = sum([ len(indexes[c]) for c in range(nb_classes) ])
        assert len(all_indexes) == min(sum(total_of_examples), max_indexes), \
           'Expected {} distinct examples but got {}'.format(\
                  sum(total_of_examples), \
                  len(all_indexes))

    return partition, val_indexes

# Number of distinct examples available 
# for training for each class
def distinct_train_set_example_size (params):
    dataset_params = params['dataset']
    dataset_name = dataset_params['name']
    nb_classes = dataset_params['nb-classes']
    validation_examples_per_class = dataset_params['validation-examples-per-class']
    train_set_number_of_examples = numbers[dataset_name]['original_train_set_number_of_examples']
    return [ train_set_number_of_examples[c] - validation_examples_per_class[c]
             for c in range(nb_classes) ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide Training and Test Data Options.')
    parser.add_argument('--rundir', type=str, default=None,
            help='Directory of the run in which to save the dataset options.')
    parser.add_argument('--name', type=str, default='mnist', choices=['mnist', 'cifar10', 'svhn'],
            help='Name of dataset for training and test. (default: mnist)')
    parser.add_argument('--train-examples-per-class', type=int, default=None, nargs='+',
            help="Number of examples to use for training for each class. " +\
                 "Examples used for validation are removed first. " +\
                 "Then if the requested number is larger than those remaining, " +\
                 "resampling is done among those." +\
                 "(default: use all available examples)")
    parser.add_argument('--validation-examples-per-class', type=int, default=None, nargs='+',
            help="Number of examples to use for validation (hyper-parameter tuning) per class. " +\
                 "The maximum number is equal to the test set size." +\
                 "(default: similar to test set.)")

    parser.add_argument('--log-partition-indexes', action='store_const', const=True, default=False, 
            help="Log the indexes of examples for each node X in ./debug/train/X.samples. ( default: False)")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    parser.add_argument('--data-directory', type=str, default=data_dir,
            help='Directory in which datasets are saved.')

    args = parser.parse_args()
    rundir = m.rundir(args)

    if args.validation_examples_per_class == None:
        args.validation_examples_per_class = [ x for x in numbers[args.name]['val_set_number_of_examples'] ]

    if args.train_examples_per_class == None:
        args.train_examples_per_class = [ (x-v) for v,x in zip(args.validation_examples_per_class, 
            numbers[args.name]['original_train_set_number_of_examples']) ]


    dataset = {
       'name': args.name,
       'train-examples-per-class': args.train_examples_per_class,
       'validation-examples-per-class': args.validation_examples_per_class,
       'data-directory': args.data_directory,
       'nb-classes': len(numbers[args.name]['classes']),
       'log-partition-indexes': args.log_partition_indexes
    }

    download(dataset) # Force download of the dataset if not already present
    validate(dataset) # Ensure input parameters are valid

    m.extend(rundir, 'dataset', dataset) # Add to run parameters

    if args.rundir is None:
        print(rundir)
