from __future__ import print_function
import os
import json
import time
import sys
import argparse
import math
import functools
import random
from random import Random
import logging
from subprocess import check_output
import pickle
import statistics
import numbers

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.multiprocessing import Process, JoinableQueue, Pipe
from resnet import ResNet
from lenet import LeNet
from gn_lenet import GN_LeNet

# *************** Datasets *******************************
Ref = {
  'mnist': {},
  'cifar10': {}
}

Ref['mnist']['input-size'] = 784

# For future reference, here are,
# the number of examples of each class in the original MNIST training set:
Ref['mnist']['original_train_set_number_of_examples'] = [
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

# the number of examples of each class in the original MNIST test set (and used for the validation set): 
Ref['mnist']['original_test_set_number_of_examples'] = [
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
Ref['mnist']['val_set_number_of_examples'] = Ref['mnist']['original_test_set_number_of_examples']

# the number of examples of each class in the final training set (taken from the original training set):
Ref['mnist']['train_set_number_of_examples'] =  [
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

Ref['cifar10']['input-size'] = 3072
Ref['cifar10']['original_train_set_number_of_examples'] = [ 5000 for _ in range(10) ] 
Ref['cifar10']['original_test_set_number_of_examples'] = [ 1000 for _ in range(10) ]
Ref['cifar10']['val_set_number_of_examples'] = Ref['cifar10']['original_test_set_number_of_examples']
Ref['cifar10']['train_set_number_of_examples'] = [ 4000 for _ in range(10) ]

def _get_image_size(img):
    if torchvision.transforms.functional._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

# The corresponding transformations of torchvision rely on random numbers but their seed can only
# be set globally. These are reimplementations that make seeding the random number generator local.
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, random=Random()):
        self.p = p
        self.random = random

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.random.random() < self.p:
            return torchvision.transforms.functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant', random=Random()):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.random = random
        RandomCrop.random = random

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = RandomCrop.random.randint(0, h - th)
        j = RandomCrop.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        F = torchvision.transforms.functional

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

# ***** Models ****
def create_model(args):
    if args.model == 'linear' or args.model == 'linear-old': 
        return Net(args)
    elif args.model == 'resnet20':
        if args.dataset == 'mnist':
            return ResNet(20, (1,28,28), 10)
        elif args.dataset == 'cifar10':
            return ResNet(20, (3, 32, 32), 10)
        else:
            print('Unsupported dataset {} for ResNet'.format(args.dataset))
            sys.exit(1)
    elif args.model == 'lenet':
        if args.dataset == 'cifar10':
            return LeNet(3,10)
        elif args.dataset == 'mnist':
            return LeNet(1,10,args=args)
        else:
            print('Unsupported dataset {} for LeNet'.format(args.dataset))
            sys.exit(1)
    elif args.model == 'gn-lenet':
        if args.dataset == 'cifar10':
            return GN_LeNet(3,10)
        elif args.dataset == 'mnist':
            return GN_LeNet(1,10,args=args)
        else:
            print('Unsupported dataset {} for GN_LeNet'.format(args.dataset))
            sys.exit(1)
    else:
        print('Unsupported model {}'.format(args.model))
        sys.exit(1)

def copy_model(model, args):
    with torch.no_grad():
        copy = create_model(args)
        for c, p in zip(copy.parameters(), model.parameters()):
            c.mul_(0.)
            c.add_(p)
        return copy

def update_model(model, new_model):
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), new_model.parameters()):
            p.mul_(0.)
            p.add_(new_p)

def model_distance(model1, model2):
    # Compute Euclidean distance (L2-norm) considering all parameters
    with torch.no_grad():
        ds = torch.tensor([
            torch.sum((p1 - p2)**2) for p1, p2 in zip(model1.parameters(), model2.parameters())
        ])
        return torch.sqrt(torch.sum(ds)).tolist()

def average_models(models, args, weights=None):
    with torch.no_grad():
        if weights == None:
            weights = [ float(1./len(models)) for _ in range(len(models)) ]
        center_model = create_model(args)
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

def update_gradients(models, gradients):
    with torch.no_grad():
        for m in models:
            for g,p in zip(gradients, m.parameters()):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                p.grad.data.zero_()
                p.grad.data.add_(g)

def format_input(data, args):
    if (args.model == 'linear' or args.model == 'linear-old') and args.dataset == 'mnist':
        return data.view(-1,Ref['mnist']['input-size']) # Turn 2D image into 1D
    elif args.model == 'linear' and args.dataset == 'cifar10':
        return data.view(-1,Ref['cifar10']['input-size']) # Turn 2D image into 1D
    elif args.model == 'resnet20' or args.model == 'lenet' or args.model == 'gn-lenet':
        return data
    else:
        print('Unsupported input with model {} and dataset {}'.format(args.model, args.dataset))
        sys.exit(1)

# ***** Nodes *****
def assign_classes (args):
    seed = args.seed
    nb_nodes = args.nb_nodes
    nodes_per_class = args.nodes_per_class
    rand = Random()
    rand.seed(seed)
    remaining_classes = [ n for n in nodes_per_class ]

    # Sample without putting the previous samples back 
    # in the bag until empty, to guarantee coverage 
    # of all classes with few nodes or rare classes
    def sampler_gen (remaining_classes, k):
        while sum(remaining_classes) > 0:
            choices = []
            for c in range(10):
                if remaining_classes[c] > 0:
                    choices.append(c)
            s = rand.sample(choices, k)
            for c in s:
                remaining_classes[c] -= 1
            yield s

    def classes():
        samples = next(sampler)
        classes = [ 0. for c in range(10) ]
        for i in range(args.local_classes):
            classes[samples[i]] = 1.
        return classes

    sampler = sampler_gen(remaining_classes, args.local_classes)
    nodes = [ { "rank": i, "classes": classes() } for i in range(nb_nodes) ]
    multiples = [ 0 for _ in range(10) ] 
    for n in nodes:
        for c in range(10):
            multiples[c] += n["classes"][c]

    logging.info('assign_classes: classes represented times {}'.format(multiples))
    return nodes 

# Ensure each node has a disjoint set of examples
# with all examples are assigned to nodes
def assign_sample_ranges(nodes, args):
    # Use the same (max) number for all classes to ensure
    # equal number of training examples regardless
    # of the class(es) chosen. Repeat examples when necessary.
    train_examples_not_used_in_val_set = [ Ref[args.dataset]['val_set_number_of_examples'][c] - 
                          math.floor(Ref[args.dataset]['val_set_number_of_examples'][c] * 
                                     args.validation_set_ratio) for c in range(10) ]

    examples_per_class = max([ Ref[args.dataset]['train_set_number_of_examples'][c] + 
                               train_examples_not_used_in_val_set[c] for c in range(10) ])

    # save [start, end[ for each class of every node where:
    # 'start' is the inclusive start index
    # 'end' is the exclusive end index
    start = [ 0 for i in range(10) ]
    for n in nodes:
        end = [ start[c] + int(math.ceil(n["classes"][c] * args.local_train_ratios[c] * examples_per_class))  
                for c in range(10) ]
        n['samples'] = [(start[c], end[c]) for c in range(10)]
        start = end

    return nodes, end

# ***** Metrics **************
def dissimilarity (n1, n2):
    total = 0
    for i in range(len(n1["classes"])):
        diff = abs(n1["classes"][i] - n2["classes"][i])
        total += diff
    return total

def similarity (n1, n2):
    return - dissimilarity(n1,n2)

def random_metric (seed):
    memory = {}
    rand = Random()
    rand.seed(seed)
    # mimic behaviour of similarity/dissimilarity:
    # - same maxValue (1 because global_prob sums to 1)
    # - same minValue (0)
    # - commutative
    # - deterministic 
    max_value = 1 
    min_value = 0

    def metric (n1, n2):
        k = str(n1['rank']) + ' ' + str(n2['rank']) if n1['rank'] > n2['rank'] else str(n2['rank']) + ' ' + str(n1['rank'])
        if k in memory:
            return memory[k]
        else:
            v = rand.uniform(min_value, max_value)
            memory[k] = v
            return v

    return metric

metrics = {
  "similarity": similarity,
  "dissimilarity": dissimilarity
}

# ***** Topologies ***********
def path(nodes, metric, position=None, rank=None, args={}):
    if position is not None or rank is not None:
        raise "Unsupported position or rank argument for path"
    remaining = [ n["rank"] for n in nodes ]
    n = remaining.pop()
    edges = {n:set()}
    while len(remaining) > 0:
        remaining.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(metric(nodes[r1], nodes[n]) - metric(nodes[r2], nodes[n])))))
        t = remaining.pop()
        edges[n] = { t } if not n in edges else edges[n].union({t})
        edges[t] = { n } if not t in edges else edges[t].union({n})
        n = t
    return edges

# position and rank arguments are added to the interface uniform 
# for topology generation. They are both ignored because they all 
# result in equivalent rings.
def ring(nodes, metric, position=None, rank=None, args={}):
    remaining = [ n["rank"] for n in nodes ]
    n = remaining.pop()
    edges = {n:set()}
    first = n
    while len(remaining) > 0:
        remaining.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(metric(nodes[r1], nodes[n]) - metric(nodes[r2], nodes[n])))))
        t = remaining.pop()
        edges[n] = { t } if not n in edges else edges[n].union({t})
        edges[t] = { n } if not t in edges else edges[t].union({n})
        n = t
    # complete the ring
    if (len(nodes) > 1):
        edges[n] = edges[n].union({first})
        edges[first] = edges[first].union({n})
    return edges

def grid(nodes, metric, position=None, rank=None, args={}):
    # Greedy algorithm: chooses the next node that maximize the metric chosen with the two neighbours of the
    # previous diagonal wavefront.
    def sort_fn (prev,curr):
        def fn (r1, r2):
            m1 = metric(nodes[r1],nodes[prev]) - metric(nodes[r2], nodes[prev]) if prev else 0.0
            m2 = metric(nodes[r2],nodes[curr]) - metric(nodes[r2], nodes[curr]) if curr else 0.0
            return int(1000.0*m1+1000.0*m2)
        return fn


    remaining = [ n["rank"] for n in nodes ]
    wavefront = [remaining.pop()] # grid diagonal on which new nodes are grafted
    edges = {wavefront[0]: set()}
    middle = math.ceil(math.sqrt(len(nodes))) # largest diagonal
    assert middle == math.sqrt(len(nodes)), "grid: unsupported non-square node number"
    next_wavefront = []
    wi = 1 # wavefront index

    while len(remaining) > 0:
        if wi < middle:
            # start and end at borders, to add new nodes with single edges
            first = 0
            last = len(wavefront) + 1
        else:
            # start and end inside the borders,
            # adding only new nodes for each pair of wavefront nodes
            first = 1
            last = len(wavefront)

        for i in range(first, last):
            prev = wavefront[i-1] if i > 0 else None
            curr = wavefront[i] if i < len(wavefront) else None
            remaining.sort(key=functools.cmp_to_key(sort_fn(prev,curr)))
            t = remaining.pop()
            if prev != None:
                edges[t] = edges[t].union({ prev }) if t in edges else { prev }
                edges[prev] = edges[prev].union({ t }) if prev in edges else { t }
            if curr != None:
                edges[t] = edges[t].union({ curr }) if t in edges else { curr }
                edges[curr] = edges[curr].union({ t }) if curr in edges else { t }
            next_wavefront.append(t)

        wavefront = next_wavefront
        next_wavefront = []
        wi += 1

    return edges

# generate a grid by placing nodes in a clockwise spiral
# from an initial position
# -> enables choosing the initial position of a single node
# -> favors clusters in a box fashion (rather than diagonal)
# 
# Grid Coordinates (i,j): 
# (0,0) -> (1,0) -> ...
#   |
#   V
# (0,1)
#   | 
#   V
#  ...
def grid2(nodes, metric, position=None, rank=None, args={}):
    def out_of_bounds (i,j):
        return i < 0 or i >= side_len or j < 0 or j >= side_len

    def distance (i,j,node):
        d = 0.0
        d += metric(grid[i-1][j], node) if not out_of_bounds(i-1, j) and grid[i-1][j] != None else 0.0
        d += metric(grid[i+1][j], node) if not out_of_bounds(i+1, j) and grid[i+1][j] != None else 0.0
        d += metric(grid[i][j-1], node) if not out_of_bounds(i, j-1) and grid[i][j-1] != None else 0.0
        d += metric(grid[i][j+1], node) if not out_of_bounds(i, j+1) and grid[i][j+1] != None else 0.0
        return d

    if position is None:
        position = (0,0)

    side_len = math.ceil(math.sqrt(len(nodes)))
    assert side_len == math.sqrt(len(nodes)), "grid: unsupported non-square node number"
    assert not out_of_bounds(position[0], position[1]), "start position out of bounds"
    assert rank == None or (type(rank) == int and rank >= 0 and rank < len(nodes)), "invalid first node rank"

    # Current box limits
    left = position[0]
    right = position[0]
    top = position[1]
    bottom = position[1]
    # Current coordinates
    i = position[0]
    j = position[1]
    # Remaining nodes
    remaining = [ i for i in range(len(nodes)) if i != rank]
    # Uninitialized Grid
    grid = [ [ None for j in range(side_len) ] for i in range(side_len) ]

    # Place first node
    if rank == None:
        node = nodes[remaining.pop()]
    else:
        node = nodes[rank]
    grid[i][j] = node

    # Topology
    edges = {node["rank"]: set()}

    # Set base case
    last_i = i
    last_j = j
    last_position = 'right-top'

    while len(remaining) > 0:
        # increase box limits once the box is filled
        if i == last_i and j == last_j and last_position == 'right-top':
            right += 1
            bottom += 1
            last_position = 'left-bottom'
            last_i = left
            last_j = bottom
        if i == last_i and j == last_j and last_position == 'left-bottom':
            left -= 1
            top -= 1
            last_position = 'right-top'
            last_i = right
            last_j = top

        # pick next empty spot in a spiral pattern
        if i < right and j == top:
            i += 1
        elif j < bottom and i == right:
            j += 1
        elif i > left and j == bottom:
            i -= 1
        elif j > top and i == left:
            j -= 1

        # skip spots outside of grid limits
        if out_of_bounds(i,j):
            continue

        # find best next node, according to the metric evaluated
        # with previously assigned neighbours
        distances = { rank:distance(i,j,nodes[rank]) for rank in remaining }
        remaining.sort(key=functools.cmp_to_key(lambda a, b: distances[a] - distances[b]))

        # place node
        rank = remaining.pop() # best node
        grid[i][j] = nodes[rank]

        # assign edges
        for (ii,jj) in [(i,j-1), (i,j+1), (i+1,j), (i-1,j)]:
            if not out_of_bounds(ii, jj) and grid[ii][jj] != None:
                neighbour = grid[ii][jj]["rank"]
                edges[rank] = edges[rank].union({neighbour}) if rank in edges else {neighbour}
                edges[neighbour] = edges[neighbour].union({rank}) if neighbour in edges else {rank} 

    return edges

# position and rank are ignored because they result in equivalent
# topologies
def fully_connected(nodes, metric, position=None, rank=None, args={}):
    all_ranks = { n["rank"] for n in nodes }
    return { x: all_ranks.difference({x}) for x in all_ranks }

# Generate grid as usual but with probability <= args.sm_beta
# for each edge, rewire to a randomly chosen node in the network,
# excluding self-connections and repeated connections
# For explanations see: Watts, Small Worlds, p67
# or https://doi.org/10.1038/30918
def small_world_grid(nodes, metric, position=None, rank=None, args={}, tries=100):
    rand = Random() 
    rand.seed(args.seed)

    for _ in range(tries):
        edges = grid2(nodes, metric, position, rank, args)
        done = False
        tovisit = [ list(edges[rank]) for rank in range(len(nodes)) ]
        while not done:
            done = True
            for rank in range(len(nodes)):
                es = tovisit[rank]
                if len(es) > 0:
                    done = False

                    # remove both sides from tovisit
                    print(es)
                    n = es.pop()
                    print(tovisit[n])
                    tovisit[n].remove(rank)

                    # rewire the edge with probability 'sm-beta'
                    if rand.uniform(0,1.) < args.sm_beta:
                        # remove original edge
                        edges[rank] = edges[rank].difference({n})
                        edges[n] = edges[n].difference({rank})

                        # pick a new neighbour that is not
                        # ourselves and that is not already our
                        # neighbour
                        n = rank
                        while n == rank or n in edges[rank]:
                            n = rand.sample(range(0,len(nodes)), 1)[0]

                        # add back the rewired edge
                        edges[rank] = edges[rank].union({n})
                        edges[n] = edges[n].union({rank})

        # ensure the graph is connected
        tovisit = [ n for n in range(len(nodes)) ]
        seen = [0]
        while len(seen) > 0:
            current = seen.pop()
            # add neighbours
            for n in edges[current]:
                if n in tovisit:
                    tovisit.remove(n)
                    seen.append(n)
        if len(tovisit) == 0:
            return edges
    raise Exception("sm-grid: internal error, did not find connected graph after {} tries.".format(tries))

def disconnected_cliques(nodes, metric, position=None, rank=None, clique_size=10, args={}):
    remaining = [ n["rank"] for n in nodes ]
    edges = {}
    def distance(n, clique):
        if len(clique) == 0:
            return 0
        return sum([ metric(nodes[n], nodes[m]) for m in clique ])

    cliques = []
    while len(remaining) > 0:
        clique = []
        for _ in range(clique_size):
            if len(remaining) == 0:
                break

            remaining.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(distance(r1, clique) - distance(r2, clique)))))
            n = remaining.pop()
            edges[n] = set(clique)
            for m in clique:
                edges[m] = { n } if not m in edges else edges[m].union({n})
            clique.append(n)
        cliques.append(clique)

    args.cliques = [ c.copy() for c in cliques ]

    seed = args.seed
    rand = Random()
    rand.seed(seed)
    for clique in cliques:
        candidates = [] 
        for i in range(len(clique)-1):
            for j in range(i+1, len(clique)):
                candidates.append((clique[i],clique[j]))
        rand.shuffle(candidates)
        for (m,n) in candidates[:args.remove_clique_edges]:
            edges[m].remove(n)
            edges[n].remove(m)

    if args.unbiased_gradient:
        args.averaging_neighbourhood = [ list(edges[n['rank']].union({n['rank']})) for n in nodes ]

    if args.print_track_cliques:
        logging.info('to track cliques, add options: {}'.format(' '.join([ "--track-cluster '{}'".format(' '.join(map(str,clique))) for clique in cliques ])))
        sys.exit(0)

    for edge in args.add_edge:
        m = edge[0]
        n = edge[1]
        assert m != n, 'Invalid edge between {} and itself'.format(m)
        edges[m] = { n } if not m in edges else edges[m].union({n})
        edges[n] = { m } if not n in edges else edges[n].union({m})

    return edges

def clique_ring(nodes, metric, position=None, rank=None, clique_size=10, args={}):
    edges = disconnected_cliques(nodes, metric, position, rank, clique_size, args) 
    cliques = [ set(c) for c in args.cliques ]

    # create the ring
    prev = cliques[-1].pop()
    for clique in cliques:
        current = clique.pop()
        # add edge between current and prev
        edges[prev].add(current)
        edges[current].add(prev)
        # prepare for next
        prev = clique.pop()
    return edges

def fractal_cliques(nodes, metric, position=None, rank=None, clique_size=10, args={}):
    edges = disconnected_cliques(nodes, metric, position, rank, clique_size, args) 
    cliques = [ { m:0 for m in c } for c in args.cliques ]

    seed = args.seed
    rand = Random()
    rand.seed(seed)

    def connect(cliques):
        def least_connected_rand(clique):
            m = min(clique.values())
            lc = [ n for n in clique.keys() if clique[n] <= m ]
            rand.shuffle(lc)
            return lc

        # connect cliques from different nodes
        for i in range(len(cliques)-1):
            for j in range(i+1,len(cliques)):
                x = least_connected_rand(cliques[i]).pop()
                cliques[i][x] += 1
                y = least_connected_rand(cliques[j]).pop()
                cliques[j][y] += 1
                edges[x].add(y)
                edges[y].add(x)

        merged = {}
        for c in cliques:
            merged.update(c)
        return merged

    while len(cliques) > 1:
        toconnect = cliques.copy()
        cliques = [ connect(toconnect[i:i+10]) for i in range(0,len(toconnect),clique_size) ]

    return edges

def fully_connected_cliques(nodes, metric, position=None, rank=None, clique_size=10, args={}):
    edges = disconnected_cliques(nodes, metric, position, rank, clique_size, args) 
    cliques = [ { m:0 for m in c } for c in args.cliques ]

    def least_connected(clique):
        m = min(clique.values())
        lc = [ n for n in clique.keys() if clique[n] <= m ]
        return lc

    # connect cliques from different nodes
    for i in range(len(cliques)-1):
        for j in range(i+1,len(cliques)):
            x = least_connected(cliques[i]).pop()
            cliques[i][x] += 1
            y = least_connected(cliques[j]).pop()
            cliques[j][y] += 1
            edges[x].add(y)
            edges[y].add(x)
    return edges

def random_ten(nodes, metric, position=None, rank=None, args={}):
    seed = args.seed
    rand = Random()
    rand.seed(seed)

    found = False
    while not found:  
        edges = { n['rank']: set() for n in nodes }

        for n in nodes:
            rank = n['rank']
            available = [ m['rank'] for m in nodes 
                          if m['rank'] != rank
                          and len(edges[m['rank']]) < 10 
                          and m['rank'] not in edges[rank] ]
            rand.shuffle(available)
            toadd = (10 - len(edges[rank]))
            for neighbour in available[:toadd]:
                edges[rank].add(neighbour)
                edges[neighbour].add(rank)

        found = True
        for n in nodes:
            if len(edges[n['rank']]) != 10:
                found = False
                break
        if not found:
            logging.info('random_ten: current solution invalid, trying another one') 
    return edges

def greedy_diverse_ten(nodes, metric, position=None, rank=None, args={}):
    metric = dissimilarity
    # Even if neighbours don't necessarily have edges among themselves,
    # this distance metric, when used with 'dissimilarity' ensures a 
    # representation of all classes.
    def distance(n, neighbours):
        if len(neighbours) == 0:
            return 0
        return sum([ metric(nodes[n], nodes[m]) for m in neighbours ])

    def class_in_neighbours(c, neighbours):
        for n in neighbours:
            if any([ int(c1) & int(c2) for c1,c2 in zip(c, nodes[n]['classes']) ]):
                return True

    edges = { n['rank']:set() for n in nodes }

    for n in nodes:
        rank = n['rank']
        available = [ m['rank'] for m in nodes 
                      if m['rank'] != rank
                      and len(edges[m['rank']]) < 10 
                      and m['rank'] not in edges[rank] 
                      and not class_in_neighbours(n['classes'], edges[m['rank']])]

        neighbours = edges[rank].union({rank})
        toadd = 9 - len(edges[rank])
        for _ in range(toadd):
            available.sort(key=functools.cmp_to_key(lambda r1,r2: int(1000*(distance(r1, neighbours) - distance(r2, neighbours)))))
            new_neighbour = available.pop()
            neighbours.add(new_neighbour)
            edges[rank].add(new_neighbour)
            edges[new_neighbour].add(rank)

    if args.unbiased_gradient:
        args.averaging_neighbourhood = [ list(edges[n['rank']].union({n['rank']})) for n in nodes ]
    
    # Add one last edge to all nodes
    seed = args.seed
    rand = Random()
    rand.seed(seed)
    for n in nodes:
        rank = n['rank']
        available = [ m['rank'] for m in nodes 
                      if m['rank'] != rank
                      and len(edges[m['rank']]) < 10 
                      and m['rank'] not in edges[rank] ]
        rand.shuffle(available)
        toadd = (10 - len(edges[rank]))
        for neighbour in available[:toadd]:
            edges[rank].add(neighbour)
            edges[neighbour].add(rank)

    for n in nodes:
        rank = n['rank']
        assert len(edges[rank]) == 10
        classes = n['classes'].copy()
        for m in edges[n['rank']]:
            for c in range(10):
                classes[c] += nodes[m]['classes'][c]
        for c in range(10):
            assert classes[c] >= 1 and classes[c] <= 2

    return edges

# Add additional edges per node in a "smallworld" topology built
# upon a ring: cliques preferentially attach to cliques closer on
# the ring (clock-wise and counter-clockwise) with exponentially less edges
# the further away on the ring.
def smallworld_logn_cliques(nodes, metric, position=None, rank=None, clique_size=10, args=None):
    edges = disconnected_cliques(nodes, metric, position, rank, clique_size, args) 
    cliques = [ { m:0 for m in c } for c in args.cliques ]

    seed = args.seed
    rand = Random()
    rand.seed(seed)

    def least_connected(clique):
        # Pick the nodes with the smallest number of edges within
        # the smallest cliques
        m = min(clique.values())
        nodes = [ n for n in clique.keys() if clique[n] == m ]
        rand.shuffle(nodes)
        return nodes

    offsets = [ 2**s for s in range(0,math.ceil(math.log(len(cliques))/math.log(2))) ]
    for start in range(len(cliques)):
        for offset in offsets:
            for k in range(2):
                # Connect to a clique in the negative direction
                x = least_connected(cliques[start]).pop()
                cliques[start][x] += 1
                c = (start-offset-k)%len(cliques)
                y = least_connected(cliques[c]).pop()
                cliques[c][y] += 1
                edges[x].add(y)
                edges[y].add(x)

                # Connect to a clique in the positive direction
                x = least_connected(cliques[start]).pop()
                cliques[start][x] += 1
                c = (start+offset+k)%len(cliques)
                y = least_connected(cliques[c]).pop()
                cliques[c][y] += 1
                edges[x].add(y)
                edges[y].add(x)
    return edges
     

topologies = {
    "path": path,
    "ring": ring,
    "grid": grid2,
    "fully_connected": fully_connected,
    "sm-grid": small_world_grid,
    "disconnected-cliques": disconnected_cliques,
    "clique-ring": clique_ring,
    "fully-connected-cliques": fully_connected_cliques,
    "fractal-cliques": fractal_cliques,
    "random-10": random_ten,
    "greedy-diverse-10": greedy_diverse_ten,
    "smallworld-logn-cliques": smallworld_logn_cliques
}

# ***** Train Set Subset *****
def partition(rank, ranges, totals, args, val_set=None, test_set=True):
    rand = Random() 
    # Make the random number generation identical for all nodes
    # to generate the same list of examples everywhere
    rand.seed(args.seed)
    logging.info('partition: loading {} training examples'.format(args.dataset))

    if args.dataset == 'mnist':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        train = datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transform)
    elif args.dataset == 'cifar10':
        # Transforms taken from this tutorial: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
        transform = transforms.Compose([
            #transforms.Pad(4),
            #RandomHorizontalFlip(random=rand), # Reimplemented to be able to use a deterministic seed
            #RandomCrop(32, random=rand),       # Reimplemented to be able to use a deterministic seed
            transforms.ToTensor()])
        train = datasets.CIFAR10(
            '../data',
            train=True,
            download=True,
            transform=transform)
    else:
        print('Unsupported dataset {}'.format(args.dataset))
        sys.exit(1)


    logging.info('partition: extracting class indexes')
    indexes = { x: [] for x in range(10) }
    for x in indexes:
        c = (torch.tensor(train.targets) == x).nonzero()
        indexes[x] = c.view(len(c)).tolist()

    logging.info('partition: choosing validation examples')
    rand_val = Random()
    rand_val.seed(1337) # The validation set is the same for all experiments
    val_indexes = [] 
    for c in range(10):
        rand_val.shuffle(indexes[c])
        upper_val_index = math.floor(Ref[args.dataset]['val_set_number_of_examples'][c] * 
                               args.validation_set_ratio)
        val_indexes.extend(indexes[c][0:upper_val_index])
        indexes[c] = indexes[c][upper_val_index:]
        if args.validation_set_ratio == 1.0:
            assert len(indexes[c]) == Ref[args.dataset]['train_set_number_of_examples'][c],\
                "Expected train set for class {} to have {} examples instead of {}".format(
                    c, 
                    Ref[args.dataset]['train_set_number_of_examples'][c], 
                    len(indexes[c])
                )
    logging.info('partition: validation set size {}'.format(len(val_indexes)))
    
    logging.info('partition: choosing training examples')
    partition = []
    # We avoid explicitly passing and saving individual indexes of training examples
    # by using ranges over the randomized list of indexes, that is the same for all nodes.
    # 
    # 'assign_sample_ranges' generates the ranges centrally and quickly from the root process
    # then each node picks the examples covered by their range with the following method.
    #
    # for each class:
    for c in range(10):
        # Get the limits of our range
        start = ranges[c][0]
        end = ranges[c][1]
        # initialize the pseudo-random number generator specifically for this class
        rand.seed(args.seed + c)
        # Generate the randomized list of examples,
        # while ensuring all indexes are covered for each previous range of len(indexes[c])
        ind_len = len(indexes[c])
        min_len = max(ind_len,totals[c])
        rand_examples = []
        for i in range(int(math.ceil(min_len/ind_len))):
            rand_examples.extend(rand.sample(range(ind_len), ind_len))

        # Get the range specific to this node
        r = rand_examples[start:end]
        # Get the index of examples in the training set
        partition.extend([indexes[c][i] for i in r ])
    # Available examples are greater than the remaining training examples of MNIST 
    # because we repeat some examples for smaller classes to have equal training
    # set sizes, regardless of the classes represented

    train_examples_not_used_in_val_set = [ Ref[args.dataset]['val_set_number_of_examples'][c] - 
                          math.floor(Ref[args.dataset]['val_set_number_of_examples'][c] * 
                                     args.validation_set_ratio) for c in range(10) ]
    available_examples = max([ Ref[args.dataset]['train_set_number_of_examples'][c] + 
                               train_examples_not_used_in_val_set[c] for c in range(10) ]) * 10
    ratio = float(len(partition))/available_examples
    logging.info('partition: total number of training examples {} ({:.2f}% of available)'.format(len(partition), ratio*100))

    # Save the indexes chosen on the file system
    if args.log == 'PARTITION':
        print("Saving train samples in ./debug/train/{}.samples, 'cat debug/train/*\.samples | sort -n | uniq | wc -l' should give 50000 if you intend to have all training examples present.".format(rank))
        debug_file = os.path.join(os.getcwd(), "debug", "train", "{}.samples".format(rank))
        with open(debug_file, 'w') as debug:
            debug.write('\n'.join(map(str, partition)) + '\n')
            
        print("Saving validation samples in ./debug/valid/{}.samples, 'cat debug/valid/*\.samples | sort -n | uniq | wc -l' should give 10000.".format(rank))
        debug_file = os.path.join(os.getcwd(), "debug", "valid", "{}.samples".format(rank))
        with open(debug_file, 'w') as debug:
            debug.write('\n'.join(map(str, val_indexes)) + '\n')
        print("All validation sets should be identical: 'for i in ./debug/valid/*.samples; do md5 $i; done' should give the same value for all.".format(rank))

    bsz = min(args.batch_size, len(partition))
    logging.info('partition: batch_size {}'.format(bsz))

    logging.info('partition: loading {} testing examples'.format(args.dataset))
    logging.info('DataLoader train')
    train_set = [ train[x] for x in partition ]
    trainer = torch.utils.data.DataLoader(
        train_set, 
        batch_size=int(bsz), shuffle=True)
    logging.info('DataLoader valid')
    if val_set is None:
        val_set = torch.utils.data.DataLoader(
            [ train[x] for x in val_indexes ], 
            batch_size=1000, shuffle=True)

    if test_set is True:
        logging.info('DataLoader test')
        if args.dataset == 'mnist':
            test_set = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transform),
                batch_size=1000, shuffle=True)
        elif args.dataset == 'cifar10':
            test_set = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
                batch_size=1000, shuffle=False)
        else:
            print('Unsupported dataset {}'.format(args.dataset))
            sys.exit(1)
    else:
        test_set = None
    return trainer, bsz, val_set, test_set, ratio, train_set

# ***** Model ******

# Purely linear version of the convolution example of:
# https://github.com/seba-1511/dist_tuto.pth/
class Net(torch.nn.Module):
    """ Network architecture. """

    def __init__(self, args):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(Ref[args.dataset]['input-size'],10)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# ****** Optimization ******
def log_train_results(rank, epoch, batch_index, model, trainer, event_file, args, running_loss=0.):
    model.eval()
    total_loss = 0.0
    num_batches = 0.0
    correct = 0.0
    example_number = 0.0

    with torch.no_grad():
        for data, target in trainer:
            data, target = Variable(data), Variable(target)
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            num_batches += 1.0
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            example_number += target.size(0)


    logging.info('(Rank {}, Epoch {} Batch {}) train set: Average loss: {:.4f}'
        .format(rank, epoch, batch_index, total_loss/num_batches))
    with open(event_file, 'a') as events:
        events.write(json.dumps({
            "type": "accuracy",
            "data": "train",
            "rank": rank,
            "epoch": epoch,
            "batch": batch_index,
            "loss": total_loss/num_batches,
            "running_loss": running_loss,
            "accuracy": correct / example_number 
        }) + '\n')

def log_test_results(rank, epoch, batch_index, model, test_loader, name, event_file, args):
    model.eval()
    test_loss = 0
    correct = 0
    confusion = torch.zeros([10,10], dtype=torch.int32) 
    with torch.no_grad():
        for data, target in test_loader:
            output = model(format_input(data, args))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for p,c in zip(pred,target):
                confusion[p,c] += 1

    test_loss /= len(test_loader.dataset)

    if name != 'test':
        logging.info('(Rank {}, Epoch {}, Batch {}) {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            rank, epoch, batch_index, name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    with open(event_file, 'a') as events:
        events.write(json.dumps({
            "type": "accuracy",
            "data": name,
            "rank": rank,
            "epoch": epoch,
            "batch": batch_index,
            "loss": test_loss,
            "accuracy": float(correct)/float(len(test_loader.dataset)),
            "confusion": confusion.tolist()
        }) + '\n')

def log_scattering(pool, epoch, batch_index, event_file, args, deltas):
    logging.info('(Computing Global Statistics Epoch {} Batch {}) '.format(epoch, batch_index))

    if type(deltas[0]) == list:
        delta_sum = [ 0. for _ in range(len(pool)) ] 
        assert len(deltas[0]) == len(delta_sum)
        for delta in deltas:
            for i in range(len(delta)):
                delta_sum[i] += delta[i]
    else:
        delta_sum = deltas
    
    cluster_logs = []
    with torch.no_grad():
        for ranks in args.track_cluster:
            print('DEPRECATED track_cluster')
            sys.exit(1)
            #cluster = [ pool[rank] for rank in ranks ]
            #cluster_weight = [ n['model'].fc.weight for n in cluster ]
            #cluster_bias = [ n['model'].fc.bias for n in cluster ]
            #cluster_center = [ 
            #    torch.sum(torch.stack(cluster_weight), dim=0)/float(len(cluster_weight)),
            #    torch.sum(torch.stack(cluster_bias), dim=0)/float(len(cluster_bias))
            #]
#
#            # Use Euclidean distance (L2-norm) considering both weights and bias
#            cluster_distance = [ 
#                torch.sqrt(
#                    torch.sum((cluster_center[0] - n['model'].fc.weight)**2) +
#                    torch.sum((cluster_center[1] - n['model'].fc.bias)**2)
#                ).tolist()
#                for n in cluster ]
#            cluster_avg_distance = statistics.mean(cluster_distance)
#            cluster_std_distance = statistics.stdev(cluster_distance) if len(cluster_distance) > 1 else 0.
#
#            cluster_logs.append({
#                "center": [cluster_center[0].tolist(), cluster_center[1].tolist()],
#                "avg": cluster_avg_distance,
#                "std": cluster_std_distance,
#                "max": max(cluster_distance),
#                "min": min(cluster_distance)
#            })

        # Average of all parameters gives the average model (center position in parameter space)
        if args.model == 'linear-old':
            pool_weight = [ n['model'].fc.weight for n in pool ]
            pool_bias = [ n['model'].fc.bias for n in pool ]
            pool_center = [ 
                torch.sum(torch.stack(pool_weight), dim=0)/float(len(pool_weight)),
                torch.sum(torch.stack(pool_bias), dim=0)/float(len(pool_bias))
            ]
            # Compute Euclidean distance (L2-norm) considering both weights and bias
            pool_distance = [ 
                torch.sqrt(
                    torch.sum((pool_center[0] - n['model'].fc.weight)**2) +
                    torch.sum((pool_center[1] - n['model'].fc.bias)**2)
                ).tolist()
                for n in pool ]
            center = [pool_center[0].tolist(), pool_center[1].tolist()] # JSON conversion
        elif args.model == 'linear' or args.model == 'resnet20' or args.model == 'lenet' or args.model == 'gn-lenet':
            center_model = average_models([ n['model'] for n in pool ], args)
            pool_distance = [ model_distance(center_model, n['model']) for n in pool ]
            center = [ p.tolist() for p in center_model.parameters() ]
        else:
            print('Unsupported model {}'.format(args.model))
            sys.exit(1)
        pool_avg_distance = statistics.mean(pool_distance)
        pool_std_distance = statistics.stdev(pool_distance) if len(pool_distance) > 1 else 0.

    with open(event_file, 'a') as events:
        events.write(json.dumps({
            "type": "model-scattering",
            "epoch": epoch,
            "batch": batch_index,
            "distance_to_center": {
                "global": {
                    "center": center,
                    "avg": pool_avg_distance,
                    "std": pool_std_distance,
                    "max": max(pool_distance),
                    "min": min(pool_distance)
                },
                "clusters": cluster_logs
            },
            "deltas": delta_sum
        }) + '\n')


# Pytorch AllReduce
def allreduce(rank, nodes, edges, event_dir, meta, args):
    def average_gradients(model, train_ratio):
        """ Gradient averaging. """
        for param in model.parameters():
            param.grad.data *= train_ratio
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
    train_set, bsz, val_set, test_set, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args)
    device = "cpu"
    model = create_model(args)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

    log_test_results(dist.get_rank(), 0, 0, model, test_set, "test", event_file, args)
    log_test_results(dist.get_rank(), 0, 0, model, val_set, "valid", event_file, args)

    logging.info("allreduce: train_set ratio: {}".format(train_ratio))
    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        model.train()
        for data, target in train_set:
            batch_index += 1
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()
            average_gradients(model, train_ratio)
            optimizer.step()

        model.eval()
        log_train_results(rank, epoch, batch_index, model, train_set, event_file, args) 
        log_test_results(dist.get_rank(), epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(dist.get_rank(), epoch, batch_index, model, val_set, "valid", event_file, args)

def compute_weights(nodes, edges, args):
    # TODO implement: https://web.stanford.edu/~boyd/papers/pdf/fastavg.pdf

    weights = torch.zeros((len(nodes), len(nodes)))
    if args.topology_weights == 'equal-clique-probability':
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue # Value derived from other values
                elif j in edges[i]:
                    weights[i,j] = edges[i][j]

        for i in range(len(nodes)):
            weights[i,i] = 1. - weights[i,:].sum()
    elif args.topology_weights == 'metropolis-hasting':
        # using Metropolis Hasting (http://www.web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf, eq 4)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue # Do it last
                elif j in edges[i]:
                    weights[i,j] = 1./(max(len(edges[i]), len(edges[j])) + 1)

        for i in range(len(nodes)):
            weights[i,i] = 1. - weights[i,:].sum()
    else:
        raise Exception("Unimplemented topology_weights {}".format(args.topology_weights))
    eps = torch.finfo(torch.float32).eps
    assert all((weights.sum(axis=0) - 1.0) < 3*eps), "Weights should sum to 1. Sum is off by {}".format(weights.sum(axis=0) - 1.0)
    assert all((weights.sum(axis=1) - 1.0) < 3*eps), "Weights should sum to 1. Sum is off by {}".format(weights.sum(axis=1) - 1.0)

    return weights


# Stochastic Gradient Push: Stochastic Gradient Descent + Distributed Averaging (PushSum)
def sgp_average_model(rank, epoch, batch_index, edges, model, sgp_weight):
    es = edges[rank]
    mix = 1./(len(es)+1.) # Send the same fraction to every peer and ourselves
    # the updated parameters of model and sgp_weight represent
    # an implicit send to self
    with torch.no_grad():
        # rebias the model
        model.fc.weight *= sgp_weight if sgp_weight.numel() == 1 else sgp_weight.view(10,1)
        model.fc.bias *= sgp_weight
        # mix
        model.fc.weight *= mix
        model.fc.bias *= mix
        sgp_weight *= mix
    logging.debug("SGP Sync Node {}, Epoch {}, Batch Index {}, Mix {}".format(rank, epoch, batch_index, mix))

    # send parameters
    out_weight = model.fc.weight.clone().detach_()
    out_bias = model.fc.bias.clone().detach_() 
    out_sgp_weight = sgp_weight.clone().detach_()
    reqs = []
    for dst in es:
        logging.debug("Node {}, Epoch {}, Sending Parameters to {} (type {})".format(rank, epoch, dst, type(dst)))
        reqs.append(dist.isend(out_weight, dst))
        reqs.append(dist.isend(out_bias, dst))
        reqs.append(dist.isend(out_sgp_weight, dst))
        logging.debug("Node {}, Epoch {}, Sent SGP Bias {} to {}".format(rank, epoch, out_sgp_weight, dst))

    # receive parameters and update state
    for src in es:
        src_weight = torch.zeros_like(model.fc.weight)
        src_bias = torch.zeros_like(model.fc.bias)
        src_sgp_weight = torch.zeros_like(sgp_weight)
        with torch.no_grad():
            logging.debug("Node {}, Epoch {}, Receiving Parameters from {}".format(rank, epoch, src))
            dist.recv(src_weight, src)
            model.fc.weight += src_weight
            dist.recv(src_bias, src)
            model.fc.bias += src_bias
            dist.recv(src_sgp_weight, src)
            sgp_weight += src_sgp_weight
            logging.debug("Node {}, Epoch {}, Received SGP Bias {} from {}".format(rank, epoch, src_sgp_weight, src))

    # unbias the parameters with sgp_weight
    with torch.no_grad():
        logging.debug("Node {}, Epoch {}, Batch {}, Recovering Unbiased Average Parameters (unbias weight: {})".format(rank, epoch, batch_index, sgp_weight))
        model.fc.weight /= sgp_weight if sgp_weight.numel() == 1 else sgp_weight.view(10,1)
        model.fc.bias /= sgp_weight

    # send barrier
    for req in reqs:
        req.wait()

def sgp(rank, nodes, edges, event_dir, meta, args):
    event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
    train_set, bsz, val_set, test_set, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args)
    device = "cpu"
    model = create_model(args)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

    logging.info("sgp: train_set ratio: {}".format(train_ratio))

    if args.dist_optimization == 'sgp':
        sgp_weight = torch.tensor([1.])
        logging.info('sgp: initializing distributed averaging bias to [1.]')
    elif args.dist_optimization == 'mc-sgp':
        # Proportion of examples of each class * len(nodes) 
        # +0.01 (to avoid a division by 0 when some classes are absent)
        sgp_weight = torch.tensor([ r[1] - r[0] for r in nodes[rank]['samples'] ]) * len(nodes) + 0.01
        logging.info('mc-sgp: initializing distributed averaging bias to {}'.format(sgp_weight))
    else:
        raise Error('sgp: invalid choice of distributed optimization {}'.format(args.dist_optimization))

    log_test_results(rank, 0, 0, model, test_set, "test", event_file, args)
    log_test_results(rank, 0, 0, model, val_set, "valid", event_file, args)

    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        model.train()
        for data, target in train_set:
            # local training step
            batch_index += 1
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # distributed averaging with neighbours
            for _ in range(args.sync_per_mini_batch): 
                sgp_average_model(rank, epoch, batch_index, edges, model, sgp_weight)

        model.eval()
        log_train_results(rank, epoch, batch_index, model, train_set, event_file, args) 
        log_test_results(rank, epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(rank, epoch, batch_index, model, val_set, "valid", event_file, args)

def sgp_average_model_single_process(pool, edges):
    model_0 = pool[0]['model']
    averaged = [ [
        torch.zeros_like(model_0.fc.weight), 
        torch.zeros_like(model_0.fc.bias),
        torch.zeros_like(pool[0]['sgp_weight'])
    ] for _ in pool ]

    with torch.no_grad():
        # Compute averages
        for n in pool:
            rank = n['rank']
            es = edges[rank]
            mix = 1./(len(es)+1.) # Send the same fraction to every peer and ourselves

            # Rebias the model
            model = n['model']
            sgp_weight = n['sgp_weight']
            model.fc.weight *= sgp_weight if sgp_weight.numel() == 1 else sgp_weight.view(10,1)
            model.fc.bias *= sgp_weight

            # Set fraction of model to send to everyone including ourselves
            model.fc.weight *= mix
            model.fc.bias *= mix
            sgp_weight *= mix

        for n in pool:
            rank = n['rank']
            model = n['model']
            sgp_weight = n['sgp_weight']

            # Add contribution of self
            averaged[rank][0] += model.fc.weight
            averaged[rank][1] += model.fc.bias
            averaged[rank][2] += sgp_weight

            # Add contributions of neighbours 
            neighbours = edges[rank]
            for src in neighbours:
                averaged[rank][0] += pool[src]['model'].fc.weight
                averaged[rank][1] += pool[src]['model'].fc.bias
                averaged[rank][2] += pool[src]['sgp_weight']

        # Update models
        for n in pool:
            rank = n['rank']
            model = pool[rank]['model']
            sgp_weight = pool[rank]['sgp_weight']

            model.fc.weight *= 0.
            model.fc.bias *= 0.
            sgp_weight *= 0.
            model.fc.weight += averaged[rank][0]
            model.fc.bias += averaged[rank][1]
            sgp_weight += averaged[rank][2]

            # Unbias the parameters
            model.fc.weight /= sgp_weight if sgp_weight.numel() == 1 else sgp_weight.view(10,1)
            model.fc.bias /= sgp_weight

# See comment on d_psgd_single_process
def sgp_single_process(nodes, edges, event_dir, meta, args, logging_tasks):
    # Initialize the pool of nodes
    pool = []
    val_set = None # Speed up initialization by avoiding reloading the validation set every time
    for rank in range(len(nodes)):
        torch.manual_seed(args.seed+rank) # To have the same initialization behaviour as the multi-process version
        torchvision.utils.torch.manual_seed(args.seed+rank)
        event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
        train_set, bsz, val_set, _, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args, val_set=val_set, test_set=False)
        device = "cpu"
        model = create_model(args)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

        if args.dist_optimization == 'sgp':
            sgp_weight = torch.tensor([1.])
            logging.info('sgp: initializing distributed averaging bias to [1.]')
        elif args.dist_optimization == 'mc-sgp':
            # Proportion of examples of each class * len(nodes) 
            # +0.01 (to avoid a division by 0 when some classes are absent)
            sgp_weight = torch.tensor([ r[1] - r[0] for r in nodes[rank]['samples'] ]) * len(nodes) + 0.01
            logging.info('mc-sgp: initializing distributed averaging bias to {}'.format(sgp_weight))
        else:
            raise Error('sgp: invalid choice of distributed optimization {}'.format(args.dist_optimization))

        pool.append({
          'rank': rank,
          'event_file': event_file,
          'train_set': train_set,
          'bsz': bsz,
          'train_ratio': train_ratio,
          'model': model,
          'optimizer': optimizer,
          'sgp_weight': sgp_weight
        })
        logging.info("sgp: train_set ratio (rank: {}): {}".format(rank,train_ratio))

    def node_epoch(train_set, optimizer, model):
        model.train()
        for data, target in train_set:
            # local training step
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            yield loss

    # Create empty files first to avoid race condition on first write
    for n in pool:
        open(n['event_file'], 'a').close()

    for n in pool:
        logging_tasks.put((n['rank'], 0, 0, pickle.dumps(n['model'].state_dict()), n['event_file']))

    logging_tasks.join() # Wait for evaluation of models to complete 
    logging.info('starting training')

    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        logging.info("sgp: epoch {}".format(epoch))
        for step in zip(*[node_epoch(pool[rank]['train_set'], pool[rank]['optimizer'], pool[rank]['model']) for rank in range(len(nodes))]):
            batch_index += 1
            for _ in range(args.sync_per_mini_batch): 
                sgp_average_model_single_process(pool, edges)

        logging_tasks.join() # Wait for evaluation of models to complete 
        for n in pool:
            n['model'].eval()
            log_train_results(n['rank'], epoch, batch_index, n['model'], n['train_set'], n['event_file'], args) 

        for n in pool:
            logging_tasks.put((n['rank'], epoch, batch_index, pickle.dumps(n['model'].state_dict()), n['event_file']))

def d_psgd_average_model(rank, epoch, batch_index, edges, model, weights):
    es = edges[rank]

    # send parameters
    out_weight = model.fc.weight.clone().detach_()
    out_bias = model.fc.bias.clone().detach_() 
    reqs = []
    for dst in es:
        logging.debug("Node {}, Epoch {}, Sending Parameters to {} (type {})".format(rank, epoch, dst, type(dst)))
        with torch.no_grad():
            reqs.append(dist.isend(out_weight*weights[rank,dst], dst))
            reqs.append(dist.isend(out_bias*weights[rank,dst], dst))
        logging.debug("Node {}, Epoch {}, Sent to {}".format(rank, epoch, dst))

    # receive parameters and update state
    with torch.no_grad():
        model.fc.weight *= weights[rank,rank]
        model.fc.bias *= weights[rank,rank]

    for src in es:
        src_weight = torch.zeros_like(model.fc.weight)
        src_bias = torch.zeros_like(model.fc.bias)
        with torch.no_grad():
            logging.debug("Node {}, Epoch {}, Receiving Parameters from {}".format(rank, epoch, src))
            dist.recv(src_weight, src)
            model.fc.weight += src_weight
            dist.recv(src_bias, src)
            model.fc.bias += src_bias
            logging.debug("Node {}, Epoch {}, Received from {}".format(rank, epoch, src))

    # send barrier
    for req in reqs:
        req.wait()

def d_psgd(rank, nodes, edges, event_dir, meta, args):
    event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
    train_set, bsz, val_set, test_set, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args)
    device = "cpu"
    model = create_model(args)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

    logging.info("d_psgd: train_set ratio: {}".format(train_ratio))

    # Compute Weights
    # using Metropolis Hasting (http://www.web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf, eq 4)
    weights = torch.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue # Do it last
            elif j in edges[i]:
                weights[i,j] = 1./(max(len(edges[i]), len(edges[j])) + 1)

    for i in range(len(nodes)):
        weights[i,i] = 1. - weights[i,:].sum()

    log_test_results(rank, 0, 0, model, test_set, "test", event_file, args)
    log_test_results(rank, 0, 0, model, val_set, "valid", event_file, args)

    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        model.train()
        for data, target in train_set:
            # local training step
            batch_index += 1
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # distributed averaging with neighbours
            for _ in range(args.sync_per_mini_batch): 
                d_psgd_average_model(rank, epoch, batch_index, edges, model, weights)

        model.eval()
        log_train_results(rank, epoch, batch_index, model, train_set, event_file, args) 
        log_test_results(rank, epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(rank, epoch, batch_index, model, val_set, "valid", event_file, args)

def d_psgd_unbiased_gradient_single_process(pool, args):
    with torch.no_grad():
        if args.clique_gradient:
            for clique in args.cliques:
                models = [ pool[rank]['model'] for rank in clique ]
                clique_gradients = average_gradients(models)
                update_gradients(models, clique_gradients)
                for rank in clique:
                    pool[rank]['optimizer'].step()
        elif args.unbiased_gradient:
            for n in pool:
                rank = n['rank']
                models = [ pool[m]['model'] for m in args.averaging_neighbourhood[rank]]
                gradients = average_gradients(models)
                update_gradients([n['model']], gradients)
                pool[rank]['optimizer'].step()
        else:
            raise Exception("Incorrect call to 'd_psgd_unbiased_gradient_single_process', none of --clique-gradient or --unbiased-gradient  options provided")

def d_psgd_average_model_single_process(pool, edges, weights):
    with torch.no_grad():
        averaged = [ None for _ in pool ]

        # Compute averages
        for n in pool:
            rank = n['rank']
            models = [ n['model'] ] + [ pool[src]['model'] for src in edges[rank] ] 
            _weights = [ weights[rank,rank] ] + [ weights[src,rank] for src in edges[rank] ]
            averaged[rank] = average_models(models, args, _weights) 

        # Update models
        for n in pool:
            rank = n['rank']
            update_model(n['model'], averaged[rank])

# This single process version is necessary to scale beyond ~150 nodes,
# otherwise Pytorch runs out of addresses to assign. 
def d_psgd_single_process(nodes, edges, event_dir, meta, args, logging_tasks):
    logging.info('starting d_psgd_single_process')
    weights = compute_weights(nodes, edges, args)

    # Initialize the pool of nodes
    global_file = os.path.join(event_dir, "global.jsonlines")
    pool = []
    val_set = None # Speed up initialization by avoiding reloading the validation set every time
    for rank in range(len(nodes)):
        logging.info('creating node {}'.format(rank))
        torch.manual_seed(args.seed+rank) # To have the same initialization behaviour as the multi-process version
        torchvision.utils.torch.manual_seed(args.seed+rank)
        event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
        trainer, bsz, val_set, _, train_ratio, train_set = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args, val_set=val_set, test_set=False)
        device = "cpu"
        model = create_model(args)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)
        pool.append({
          'rank': rank,
          'event_file': event_file,
          'train_set': train_set,
          'trainer': trainer,
          'bsz': bsz,
          'train_ratio': train_ratio,
          'model': model,
          'optimizer': optimizer
        })
        logging.info("d_psgd: train_set ratio (rank: {}): {}".format(rank,train_ratio))

    def node_epoch(train_set, optimizer, model, epoch):
        lrs = [ args.learning_rate * args.learning_rate_schedule[e] for e in sorted(args.learning_rate_schedule.keys()) if epoch >= e ]
        new_lr = lrs[-1]
        logging.info('epoch {} lr {}'.format(epoch,new_lr))
        for p in optimizer.param_groups:
            p['lr'] = new_lr

        bszs = [ args.batch_size_schedule[e] for e in sorted(args.batch_size_schedule.keys()) if epoch >= e ]
        new_bsz = bszs[-1]
        logging.info('epoch {} bsz {}'.format(epoch,new_bsz))

        trainer = torch.utils.data.DataLoader(
            train_set, 
            batch_size=int(new_bsz), shuffle=True)

        model.train()
        for data, target in trainer:
            previous = copy_model(model, args)
            # local training step
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()
            
            if not args.clique_gradient and not args.unbiased_gradient:
                optimizer.step()

            yield (loss, model_distance(previous, model))

    # Create empty files first to avoid race condition on first write
    for n in pool:
        open(n['event_file'], 'a').close()

    if args.initial_averaging:
        logging.info('d_psgd_single_process: averaging initial models')
        avg_model = average_models([ pool[rank]['model'] for rank in range(len(pool)) ], args)
        for rank in range(len(pool)):
            update_model(pool[rank]['model'], avg_model)

    for n in pool:
        logging_tasks.put((n['rank'], 0, 0, pickle.dumps(n['model'].state_dict()), n['event_file']))

    logging_tasks.join() # Wait for evaluation of models to complete
    logging.info('starting training')

    batch_index = 0
    deltas = [[ 0. for _ in range(len(pool)) ]]
    log_scattering(pool, 0, batch_index, global_file, args, deltas)


    for epoch in range(1, args.nb_epochs+1):
        deltas = []
        logging.info("d_psgd: epoch {}".format(epoch))

        running_loss = [ 0. for _ in range(len(pool)) ] 
        last_batch_index = batch_index

        for step in zip(*[node_epoch(n['train_set'], n['optimizer'], n['model'], epoch) for n in pool]):
            deltas.append([ status[1] for status in step ])
            batch_index += 1

            if args.clique_gradient or args.unbiased_gradient:
                d_psgd_unbiased_gradient_single_process(pool, args)

            for _ in range(args.sync_per_mini_batch): 
                d_psgd_average_model_single_process(pool, edges, weights)

            for i,status in enumerate(step, 0):
                running_loss[i] += status[0]


        logging_tasks.join() # Wait for evaluation of models to complete
        for n in pool:
            n['model'].eval()
            n_running_loss = (running_loss[n['rank']]/(batch_index - last_batch_index)).item()
            logging.info('(Rank {}, Epoch {}, Batch {}) running loss {}'.format(n['rank'], epoch, batch_index, n_running_loss))
            log_train_results(n['rank'], epoch, batch_index, n['model'], n['trainer'], n['event_file'], args, running_loss=n_running_loss) 

        if epoch % args.accuracy_logging_interval == 0:
            for n in pool:
                logging_tasks.put((n['rank'], epoch, batch_index, pickle.dumps(n['model'].state_dict()), n['event_file']))

        log_scattering(pool, epoch, batch_index, global_file, args, deltas)

def d_psgd_worker (p, socket, nodes, event_dir, meta, args):
    _, _, val_set, test_set, _, _ = partition(0, nodes[0]['samples'], meta['total-of-examples'], args)
    context = {
        'val_set': val_set, # Speed up initialization by avoiding reloading the validation set every time,
        'local_nodes': {},
        'test_set': test_set
    }

    def init (rank):
        logging.info('Worker {} creating node {}'.format(p, rank))
        torch.manual_seed(args.seed+rank) # To have the same initialization behaviour as the other versions
        torchvision.utils.torch.manual_seed(args.seed+rank)

        event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
        trainer, bsz, context['val_set'], _, train_ratio, train_set = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args, val_set=context['val_set'], test_set=False)
        model = create_model(args)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)
        context['local_nodes'][rank] = {
          'rank': rank,
          'event_file': event_file,
          'train_set': train_set,
          'trainer': trainer,
          'bsz': bsz,
          'train_ratio': train_ratio,
          'model': model,
          'optimizer': optimizer,
          'status': 'ready',
          'running_loss': 0,
          'delta': 0,
          'last_batch_index': 0,
          'last_epoch': -1,
          'trainer_iter': None
        }
        logging.info("d_psgd: train_set ratio (rank: {}): {}".format(rank,train_ratio))
        socket.send(('ready', (rank,)))
        return False

    def update (rank, state_dict):
        context['local_nodes'][rank]['model'].load_state_dict(pickle.loads(state_dict))
        return False
    
    def log (rank, epoch, batch_index):
        node = context['local_nodes'][rank]
        node['model'].eval()
        model = node['model']
        event_file = node['event_file']
        if batch_index > 0:
            n_running_loss = (node['running_loss']/(batch_index - node['last_batch_index'])).item()
            node['last_batch_index'] = batch_index
            logging.info('(Rank {}, Epoch {} Batch {}) running loss {}'.format(rank, epoch, batch_index, n_running_loss))
            log_train_results(rank, epoch, batch_index, model, node['trainer'], node['event_file'], args, running_loss=n_running_loss) 
        if epoch % args.accuracy_logging_interval == 0:
            log_test_results(rank, epoch, batch_index, model, context['test_set'], "test", event_file, args)
            log_test_results(rank, epoch, batch_index, model, context['val_set'], "valid", event_file, args)
        return False

    def train (rank, epoch, batch):
        node = context['local_nodes'][rank]
        model = node['model']
        optimizer = node['optimizer']
        trainer = node['trainer']

        model.train()
        if epoch > node['last_epoch']:
            node['last_epoch'] = epoch

            lrs = [ args.learning_rate * args.learning_rate_schedule[e] for e in sorted(args.learning_rate_schedule.keys()) if epoch >= e ]
            new_lr = lrs[-1]
            logging.info('epoch {} lr {}'.format(epoch,new_lr))
            for p in optimizer.param_groups:
                p['lr'] = new_lr

            bszs = [ args.batch_size_schedule[e] for e in sorted(args.batch_size_schedule.keys()) if epoch >= e ]
            new_bsz = bszs[-1]
            logging.info('epoch {} bsz {}'.format(epoch,new_bsz))

            trainer = torch.utils.data.DataLoader(
                node['train_set'], 
                batch_size=int(new_bsz), shuffle=True)
            node['trainer'] = trainer
            node['trainer_iter'] = iter(trainer)

            node['running_loss'] = 0
            node['delta'] = 0

        try:
            data, target = node['trainer_iter'].next()
            previous = copy_model(model, args)
            # local training step
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()

            if not args.clique_gradient and not args.unbiased_gradient:
                optimizer.step()

            node['running_loss'] += loss
            node['delta'] += model_distance(previous, model)
            socket.send(('trained', (rank, 
                                     pickle.dumps(model.state_dict()), 
                                     pickle.dumps(previous.state_dict()),
                                     pickle.dumps(average_gradients([model])))))
        except StopIteration:
            socket.send(('epoch_ended', (rank,node['delta'])))

        return False

    def stop (rank):
        context['local_nodes'][rank]['status'] = 'done'
        local_nodes = context['local_nodes']
        if all([ local_nodes[rank]['status'] == 'done' for rank in local_nodes]):
            logging.info('Worker {} closing socket'.format(p))
            socket.close()
            return True
        else:
            return False

    def loop (handlers):
        while True:
            try:
                msg_type, args = socket.recv()
                if msg_type in handlers:
                    logging.debug("Process {} Received {} {}".format(p, msg_type, args))
                    if handlers[msg_type](*args):
                        logging.info('Worker {} terminating'.format(p))
                        break
                else:
                    print('Unsupported message type {}'.format(msg_type))
                    sys.exit(1)
            except EOFError as e:
                print('Process {}, connection closed'.format(p))
                break
    logging.info('Worker {} starting'.format(p))
    loop({
        'init': init,
        'update': update,
        'log': log,
        'train': train,
        'stop': stop
    })

def d_psgd_parallel_training (nodes, edges, event_dir, meta, args, sockets):
    # The main process acts as a central coordinator to synchronize the
    # different worker processes, while performing the averaging tasks and
    # gathering global metrics.  A single worker can train multiple nodes to
    # simulate larger networks while only using as many workers as there are
    # hardware cores.  The design aims for simplicity of gathering global
    # metrics, minimal message data beyond new model parameters, and
    # compatibility with Torch libraries (ex: for ... in API for data
    # sampling).
    #
    # Message Types
    #
    # Main->Workers init(rank): initialize the required state to train model of rank
    # Workers->Main ready(rank): initialization completed 
    #
    # Main->Workers update(rank, new_model): update the model of rank with new_model parameters
    #
    # Main->Workers log(rank): log the model's performance
    #
    # Main->Workers train(rank, epoch, batch): start the next training batch
    # Workers->Main trained(rank, model, previous, gradients): latest version of the model after training on a mini-batch (with previous version and gradients)
    # Workers->Main epoch_ended(rank, delta): could not train, no more batches remaining for this epoch
    #
    # Main->Workers stop(rank): stop training for model 'rank'

    def main_init (nodes, sockets):
        for rank in range(args.nb_nodes):
            nodes[rank]['connection'] =  sockets[rank]
            nodes[rank]['model'] = create_model(args)
            nodes[rank]['connection'].send(('init', (rank,)))
            nodes[rank]['delta'] = 0
            nodes[rank]['optimizer'] = optim.SGD(nodes[rank]['model'].parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

        if args.initial_averaging:
            logging.info('d_psgd_parallel_training: averaging initial models')
            avg_model = average_models([ nodes[rank]['model'] for rank in range(len(nodes)) ], args)
            for rank in range(len(nodes)):
                update_model(nodes[rank]['model'], avg_model)

        logging.info('d_psgd_parallel_training: {} workers started'.format(args.nb_workers))

    def main_average (nodes, edges, weights):
        for _ in range(args.sync_per_mini_batch): 
            d_psgd_average_model_single_process(nodes, edges, weights)

    def main_update (nodes):
        for rank in range(len(nodes)):
            nodes[rank]['connection'].send(('update', (rank, pickle.dumps(nodes[rank]['model'].state_dict()))))

    def main_log (nodes, epoch, batch_index, global_file):
        for rank in range(len(nodes)):
            nodes[rank]['connection'].send(('log', (rank,epoch, batch_index)))
        deltas = [ nodes[rank]['delta'] for rank in range(len(nodes)) ] 
        log_scattering(nodes, epoch, batch_index, global_file, args, deltas)

    def main_train (nodes,epoch, batch):
        for rank in range(len(nodes)):
            nodes[rank]['connection'].send(('train', (rank,epoch,batch)))

    def main_stop (nodes):
        for rank in range(len(nodes)):
            nodes[rank]['connection'].send(('stop', (rank,)))

    def main_wait_ready (nodes):
        # Assume that models are initialized in increasing order,
        # otherwise the recv_rank could be different
        for rank in range(len(nodes)):
            (msg_type, (recv_rank,)) = nodes[rank]['connection'].recv() 
            assert msg_type == 'ready'
            assert recv_rank == rank

    def main_wait_trained(nodes):
        msg_types = [ None for _ in range(len(nodes)) ]
        values = [ None for _ in range(len(nodes)) ]

        # Assume that models are trained in increasing order,
        # otherwise the recv_rank could be different
        for rank in range(len(nodes)):
            (msg_type, attrs) = nodes[rank]['connection'].recv() 
            msg_types[rank] = msg_type
            values[rank] = attrs
            assert attrs[0] == rank

        if all([ msg_type == 'trained' for msg_type in msg_types ]):
            for rank, _current, _previous, _gradients in values:
                if args.clique_gradient or args.unbiased_gradient:
                    model = nodes[rank]['model']
                    model.load_state_dict(pickle.loads(_previous))
                    gradients = pickle.loads(_gradients)
                    update_gradients([model], gradients)
                else:  
                    nodes[rank]['model'].load_state_dict(pickle.loads(_current))

            if args.clique_gradient or args.unbiased_gradient:
                d_psgd_unbiased_gradient_single_process(nodes, args)

            return True
        elif all([msg_type == 'epoch_ended' for msg_type in msg_types ]):
            for rank, delta in values:
                nodes[rank]['delta'] = delta
            return False
        else:
            print('Error inconsistent status between nodes: {}'.format(msg_types))
            sys.exit(1)

    logging.info('starting d_psgd_parallel_training')
    weights = compute_weights(nodes, edges, args)
    global_file = os.path.join(event_dir, "global.jsonlines")

    main_init(nodes, sockets)
    main_wait_ready(nodes)
    main_update(nodes)
    main_log(nodes, 0, 0, global_file)
    batch = 1
    for epoch in range(1, args.nb_epochs+1):
        logging.info("Main Process, Epoch {}".format(epoch))
        main_train(nodes, epoch, batch)
        while main_wait_trained(nodes):
            main_average(nodes, edges, weights)
            main_update(nodes)
            batch += 1
            main_train(nodes, epoch, batch)
        main_log(nodes, epoch, batch-1, global_file)
    main_stop(nodes)
    
def d2(rank, nodes, edges, event_dir, meta, args):
    event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
    train_set, bsz, val_set, test_set, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args)
    device = "cpu"
    model = create_model(args)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

    logging.info("d2: train_set ratio: {}".format(train_ratio))

    # Compute Weights
    # using Metropolis Hasting (http://www.web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf, eq 4)
    weights = torch.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue # Do it last
            elif j in edges[i]:
                weights[i,j] = 1./(max(len(edges[i]), len(edges[j])) + 1)

    for i in range(len(nodes)):
        weights[i,i] = 1. - weights[i,:].sum()

    log_test_results(rank, 0, 0, model, test_set, "test", event_file, args)
    log_test_results(rank, 0, 0, model, val_set, "valid", event_file, args)

    model_k_1 = None # Model in step k-1
    model_k_2 = None # Model in step k-2 
    grad_k_1 = None  # Gradients in step k-1
    grad_k_2 = None  # Gradients in step k-2

    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        model.train()
        for data, target in train_set:
            # local training step
            batch_index += 1
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()

            # Save model
            model_k_2 = model_k_1
            grad_k_2 = grad_k_1
            model_k_1 = [ p.clone().detach_() for p in model.parameters() ]
            grad_k_1 =  [ p.grad.clone().detach_() for p in model.parameters() ]

            # Update
            if batch_index > 1:
                with torch.no_grad():
                    for p,p2,g1,g2 in zip(model.parameters(), model_k_2, grad_k_1, grad_k_2):
                        p += (p - p2)
                        p -= args.learning_rate * (g1 - g2)
            else:
                with torch.no_grad():
                    for p in model.parameters():
                        p -= args.learning_rate * p.grad

            # distributed averaging with neighbours
            for _ in range(args.sync_per_mini_batch): 
                d_psgd_average_model(rank, epoch, batch_index, edges, model, weights)

        model.eval()
        log_train_results(rank, epoch, batch_index, model, train_set, event_file, args) 
        log_test_results(rank, epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(rank, epoch, batch_index, model, val_set, "valid", event_file, args)

def d2_single_process(nodes, edges, event_dir, meta, args, logging_tasks):
    # Compute Weights
    # using Metropolis Hasting (http://www.web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf, eq 4)
    weights = torch.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue # Do it last
            elif j in edges[i]:
                weights[i,j] = 1./(max(len(edges[i]), len(edges[j])) + 1)

    for i in range(len(nodes)):
        weights[i,i] = 1. - weights[i,:].sum()

    # Initialize the pool of nodes
    pool = []
    val_set = None # Speed up initialization by avoiding reloading the validation set every time
    for rank in range(len(nodes)):
        torch.manual_seed(args.seed+rank) # To have the same initialization behaviour as the multi-process version
        torchvision.utils.torch.manual_seed(args.seed+rank)
        event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
        trainer, bsz, val_set, _, train_ratio, train_set = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args, val_set=val_set, test_set=False)
        device = "cpu"
        model = create_model(args)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)
        pool.append({
          'rank': rank,
          'event_file': event_file,
          'train_set': train_set,
          'trainer': trainer,
          'bsz': bsz,
          'train_ratio': train_ratio,
          'model': model,
          'model_k_1': None,  # Model in step k-1
          'model_k_2': None,  # Model in step k-2 
          'grad_k_1': None,   # Gradients in step k-1
          'grad_k_2': None,   # Gradients in step k-2
          'optimizer': optimizer
        })
        logging.info("d2: train_set ratio (rank: {}): {}".format(rank,train_ratio))

    def node_epoch(trainer, optimizer, node, batch_index):
        model = node['model']
        model.train()
        for data, target in trainer:
            # local training step
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(format_input(data, args))
            loss = F.nll_loss(output, target)
            loss.backward()

            # Save model
            node['model_k_2'] = node['model_k_1']
            node['grad_k_2'] = node['grad_k_1']
            node['model_k_1'] = [ p.clone().detach_() for p in model.parameters() ]
            node['grad_k_1'] =  [ p.grad.clone().detach_() for p in model.parameters() ]

            # Update
            if batch_index > 1:
                with torch.no_grad():
                    for p,p2,g1,g2 in zip(model.parameters(), node['model_k_2'], node['grad_k_1'], node['grad_k_2']):
                        p += (p - p2)
                        p -= args.learning_rate * (g1 - g2)
            else:
                with torch.no_grad():
                    for p in model.parameters():
                        p -= args.learning_rate * p.grad

            batch_index += 1
            yield loss

    # Create empty files first to avoid race condition on first write
    for n in pool:
        open(n['event_file'], 'a').close()

    for n in pool:
        logging_tasks.put((n['rank'], 0, 0, pickle.dumps(n['model'].state_dict()), n['event_file']))

    logging_tasks.join() # Wait for evaluation of models to complete
    logging.info('starting training')

    batch_index = 1
    for epoch in range(1, args.nb_epochs+1):
        logging.info("d2: epoch {}".format(epoch))
        for step in zip(*[node_epoch(n['trainer'], n['optimizer'], n, batch_index) for n in pool]):
            batch_index += 1
            for _ in range(args.sync_per_mini_batch): 
                d_psgd_average_model_single_process(pool, edges, weights)

        logging_tasks.join() # Wait for evaluation of models to complete
        for n in pool:
            n['model'].eval()
            log_train_results(n['rank'], epoch, batch_index, n['model'], n['trainer'], n['event_file'], args) 

        for n in pool:
            logging_tasks.put((n['rank'], epoch, batch_index, pickle.dumps(n['model'].state_dict()), n['event_file']))

def walk_step(rank, epoch, batch_index, edges, model, args):
    es = edges[rank]

    # Pick one neighbour
    rand = Random() 
    rand.seed(rank + (epoch * 100) + (args.seed * 10000) + (batch_index * 10000000))
    n = rand.randint(0, len(es))

    n_index = 0
    reqs = []
    for dst in es:
        # if the model is on this node, send to chosen neighbour
        if model.fc.bias[0] != 0 and n_index == n:
            logging.debug("walk (Rank: {} Epoch: {}): sending model to Node {}".format(rank, epoch, dst))
            out_weight = model.fc.weight.clone().detach_()
            out_bias = model.fc.bias.clone().detach_() 
            with torch.no_grad():
                model.fc.weight *= 0.
                model.fc.bias *= 0.
        # otherwise send zeros
        else:
            out_weight = torch.zeros_like(model.fc.weight)
            out_bias = torch.zeros_like(model.fc.bias)

        with torch.no_grad():
            reqs.append(dist.isend(out_weight, dst))
            reqs.append(dist.isend(out_bias, dst))
        n_index += 1

    # receive model from neighbour, if applicable
    for src in es:
        src_weight = torch.zeros_like(model.fc.weight)
        src_bias = torch.zeros_like(model.fc.bias)
        with torch.no_grad():
            dist.recv(src_weight, src)
            model.fc.weight += src_weight
            dist.recv(src_bias, src)
            model.fc.bias += src_bias
            if src_bias[0] != 0:
                logging.debug("walk (Rank: {} Epoch: {}): received model from Node {}".format(rank, epoch, src))

    # send barrier
    for req in reqs:
        req.wait()

def walk(rank, nodes, edges, event_dir, meta, args):
    event_file = os.path.join(event_dir, "{}.jsonlines".format(rank))
    train_set, bsz, val_set, test_set, train_ratio = partition(rank, nodes[rank]['samples'], meta['total-of-examples'], args)
    device = "cpu"
    model = create_model(args)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

    logging.info("walk: train_set ratio: {}".format(train_ratio))

    # Nullify all models, except the one on Node 0
    if rank != 0:
        with torch.no_grad():
            model.fc.weight *= 0. 
            model.fc.bias *= 0.

    log_test_results(rank, 0, 0, model, test_set, "test", event_file, args)
    log_test_results(rank, 0, 0, model, val_set, "valid", event_file, args)

    batch_index = 0
    for epoch in range(1, args.nb_epochs+1):
        for data, target in train_set:
            # The model is now on this node, run 1 mini-batch
            if model.fc.bias[0] != 0:
                model.train()

                # local training step
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(format_input(data, args))
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            batch_index += 1

            # Send/Receive Model 
            walk_step(rank, epoch, batch_index, edges, model, args)

        model.eval()
        log_train_results(rank, epoch, batch_index, model, train_set, event_file, args) 
        log_test_results(rank, epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(rank, epoch, batch_index, model, val_set, "valid", event_file, args)

def log_models(logging_tasks, nodes, event_dir, meta, args):
    # We only care about the validation and test sets and they are the same for all nodes
    _, _, val_set, test_set, _, _ = partition(0, nodes[0]['samples'], meta['total-of-examples'], args)
    model = create_model(args)
    for rank, epoch, batch_index, state_dict, event_file  in iter(logging_tasks.get, 'STOP'):
        model.load_state_dict(pickle.loads(state_dict))
        logging.debug('saving results in {}'.format(event_file))
        log_test_results(rank, epoch, batch_index, model, test_set, "test", event_file, args)
        log_test_results(rank, epoch, batch_index, model, val_set, "valid", event_file, args)
        logging_tasks.task_done()
    logging.info('stopping')

# ****** Process Handling *****
def run(rank, nodes, edges, event_dir, meta, args):
    torch.manual_seed(args.seed+rank)
    torchvision.utils.torch.manual_seed(args.seed+rank)
    
    if args.single_process:
        # Start additional logging processes, used for speeding up the evalutation of the models
        processes = []
        logging.info('Starting logging processes and task queue')
        logging_tasks = JoinableQueue(maxsize=len(nodes))
        for _ in range(args.nb_logging_processes):
            p = Process(target=log_models, args=(logging_tasks, nodes, event_dir, meta, args))
            p.start()
            processes.append(p)

        if args.dist_optimization == 'sgp' or args.dist_optimization == 'mc-sgp':
            main_alg = sgp_single_process
        elif args.dist_optimization == 'd-psgd':
            main_alg = d_psgd_single_process
        elif args.dist_optimization == 'd2':
            main_alg = d2_single_process
        else:
            print('Unimplemented single process version of dist_optimization {}'.format(args.dist_optimization))
            sys.exit(1)

        # The main loop is also run in a separate process to avoid the deadlock
        # on Linux when the MNIST dataset is open both in a parent process
        # (this one) and child processes (the log_models) created later. This
        # issue still happens when the dataset is saved (torch.save) and
        # reloaded later (torch.load). When all processes are siblings this is
        # no longer an issue.
        logging.info('Starting Main process')
        main = Process(target=main_alg, args=(nodes, edges, event_dir, meta, args, logging_tasks))
        main.start()
        main.join()

        logging.info('closing logging processes')
        for _ in range(args.nb_logging_processes): 
            logging_tasks.put('STOP')
        for p in processes:
            p.join()
    elif args.parallel_training:
        if args.dist_optimization == 'd-psgd':
            main_alg = d_psgd_parallel_training
            worker_fn = d_psgd_worker
        else:
            print('Unimplemented parallel training version of dist_optimization {}'.format(args.dist_optimization))
            sys.exit(1)

        if args.nb_nodes < args.nb_workers:
            logging.info('parallel_training: too many workers, using {} workers instead'.format(args.nb_nodes))
            args.nb_workers = args.nb_nodes

        sockets = [ None for _ in range(args.nb_nodes) ]
        workers = []
        logging.info('Starting worker processes')
        for w in range(args.nb_workers):
            socket, remote_socket = Pipe()
            worker = Process(target=worker_fn, args=(w, remote_socket, nodes, event_dir, meta, args))
            worker.start()
            workers.append(worker)

            for rank in range(args.nb_nodes):
                if (rank % args.nb_workers) == w:
                    sockets[rank] = socket

        # The main loop is also run in a separate process to avoid the deadlock
        # on Linux when the MNIST dataset is open both in a parent process
        # (this one) and child processes (the log_models) created later. This
        # issue still happens when the dataset is saved (torch.save) and
        # reloaded later (torch.load). When all processes are siblings this is
        # no longer an issue.
        logging.info('Starting Main process')
        main = Process(target=main_alg, args=(nodes, edges, event_dir, meta, args, sockets))
        main.start()
        main.join()

        for worker in workers:
            worker.join()
    else:
        if args.dist_optimization == 'allreduce':
            allreduce(rank, nodes, edges, event_dir, meta, args)
        elif args.dist_optimization == 'sgp' or args.dist_optimization == 'mc-sgp':
            sgp(rank, nodes, edges, event_dir, meta, args)
        elif args.dist_optimization == 'd-psgd':
            d_psgd(rank, nodes, edges, event_dir, meta, args)
        elif args.dist_optimization == '1-walk':
            walk(rank, nodes, edges, event_dir, meta, args)
        elif args.dist_optimization == 'd2':
            d2(rank, nodes, edges, event_dir, meta, args)


def init_processes(rank, nodes, edges, event_dir, meta, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = args.network_interface
    dist.init_process_group(backend, rank=rank, world_size=len(nodes))
    fn(rank, nodes, edges, event_dir, meta, args)

def experiment(args):
    if args.dist_optimization == 'allreduce':
        args.topology = 'fully_connected'
        
    logging.basicConfig(level=getattr(logging, args.log.upper(), None))

    # Assign Classes to Nodes
    assert args.local_classes >= 1 and args.local_classes <= 10, "local-classes: should be between 1 and 10"
    nodes = assign_classes(args)
    nodes, total_of_examples = assign_sample_ranges(nodes, args)

    # Choose Metric
    metrics['random'] = random_metric(args.seed)
    metric = metrics[args.metric]

    # Generate Topology
    if args.first_node_class is not None:
        filtered = [ x for x in filter(lambda n: n['classes'][args.first_node_class] > 0., nodes) ]
        if len(filtered) < 1:
            print('Invalid first-node-class chosen, no nodes has examples of class {}'.format(args.first_node_class))
        first_node_rank = filtered[0]['rank']
    else:
        first_node_rank = None

    if len(args.first_node_position) > 0:
        first_node_position = args.first_node_position
    else:
        first_node_position = None

    if args.topology in topologies:
        edges = topologies[args.topology](nodes, metric, rank=first_node_rank, position=first_node_position, args=args)
    else:
        raise Error('Invalid topology {}'.format(args.topology))

    # Setup Results Directory
    experiment_dir = os.path.join(args.results_directory, time.strftime('%Y-%m-%d-%H:%M:%S-%Z'))
    os.makedirs(experiment_dir)
    event_dir = os.path.join(experiment_dir, 'events')
    os.makedirs(event_dir)

    # Save Experiment Parameters and Script Version
    meta = {}
    for k,v in args.__dict__.items():
        meta[k] = v
    meta["script"] = __file__
    meta["git-hash"] = check_output(['git', 'rev-parse', '--short', 'HEAD'])[:-1].decode()
    meta["total-of-examples"] = total_of_examples
    print(meta)
    with open(os.path.join(experiment_dir, 'meta.json'), 'w+') as meta_file:
        json.dump(meta, meta_file)

    # Save Nodes
    with open(os.path.join(experiment_dir, 'nodes.json'), 'w+') as nodes_file:
        json.dump(nodes, nodes_file)

    # Save Topology
    with open(os.path.join(experiment_dir, 'topology.json'), 'w+') as topology_file:
        if type(edges[0]) == set:
            json.dump({ rank: list(out) for rank, out in edges.items() }, topology_file)
        else:
            json.dump({ rank: out for rank, out in edges.items() }, topology_file)

    if args.single_process or args.parallel_training:
        run(-1, nodes, edges, event_dir, meta, args)
    else:
        # Start Processes
        processes = [] 

        for rank in range(len(nodes)):
            p = Process(target=init_processes, args=(rank, nodes, edges, event_dir, meta, args, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Dist Example')
    parser.add_argument('--nb-nodes', type=int, default=1, metavar='N',
                        help='number of nodes (default: 1)')
    parser.add_argument('--nb-epochs', type=int, default=10, metavar='N',
                        help='number of epochs (default: 10)')
    parser.add_argument('--local-classes', type=int, default=10,
                        help='number of classes represented in a single node (classes are chosen randomly). (default: 10)')
    parser.add_argument('--seed', type=int, default=1337, metavar='N',
                        help='seed for pseudo-random number generator')
    parser.add_argument('--train-ratio', type=float, default=None, metavar='N',
                        help='ratio of training examples of each class to use [0.,1.] inclusive (default: 1.),\
                              automatically derived if nodes-per-class is set.')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='size of mini-batch for training (default: 128)')
    parser.add_argument('--log', type=str, default='INFO', choices=['INFO', 'DEBUG', 'PARTITION'],
                        help='log level to use (default: INFO)')
    parser.add_argument('--dist-optimization', type=str, default='sgp', 
                        choices=['allreduce', 'sgp', 'd-psgd', 'mc-sgp', '1-walk', 'd2'],
                        help='dist-optimization (default: sgp)')
    parser.add_argument('--topology', type=str, default='path', choices=['path', 'ring', 'grid', 'fully_connected', 'sm-grid', 'disconnected-cliques', 'clique-ring', 'fully-connected-cliques', 'fractal-cliques', 'random-10', 'greedy-diverse-10', 'smallworld-logn-cliques'])
    parser.add_argument('--metric', type=str, default='random', choices=['similarity', 'dissimilarity', 'random'],
                        help="metric used to select nodes (default: random)")
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='N',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--learning-momentum', type=float, default=0.0, metavar='N',
                        help='learning momentum (default: 0.0)')
    parser.add_argument('--network-interface', type=str, default='eno1',
                        help='network interface on which to connect (default: eno1)')
    parser.add_argument('--learning-rate-list', type=float, nargs='+', default=[])
    parser.add_argument('--batch-size-list', type=int, nargs='+', default=[])
    parser.add_argument('--sync-per-mini-batch', type=int, default=1 , metavar='N',
                        help='number of rounds of distributed averaging to perform per mini-batch sample')
    parser.add_argument('--nodes-per-class', type=int, default=None, nargs='+',
                        help='number of nodes having examples of each class')
    # "HACK": local-train-ratios is added as an argument to use the args object to pass options to experiments
    #         and log the results as part of the experiment trace. We may want to make it configurable later
    #         but for now it should not be supplied externally.
    parser.add_argument('--local-train-ratios', type=float, default=None, nargs='+',
                        help='(internal) ratios of available examples to use for each class in a single node')
    parser.add_argument('--global-train-ratios', type=float, default=[1. for _ in range(10) ], nargs='+',
                        help='fractions (between [0.,1.] inclusive) of available training examples to use for each class for the entire network (default: [1., ...])')
    parser.add_argument('--single-process', action='store_const', const=True, default=False, 
                        help='run all nodes within a single process')
    parser.add_argument('--nb-logging-processes', type=int, default=8, metavar='N',
                        help='Number of parallel processes to log the accuracy of models. Only applicable with \'--single-process\' (default: 8)')
    parser.add_argument('--first-node-class', type=int, default=None,
            help='Class number of the first node placed in the topology (default: random)')
    parser.add_argument('--first-node-position', type=int, nargs='+', default=[],
            help='Position of the first node in the topology (nb of dimension is topology-dependent, e.g. 2 for a grid). (default: [0,...])')
    parser.add_argument('--sm-beta', type=float, default=0.03,
            help='Probability of rewiring an edge in a small-world (sm-*) topology. (default: 0.03)')

    script_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    parser.add_argument('--results-directory', type=str, default=results_dir,
                        help='directory in which to save the experiment(s) results. (default: {})'.format(results_dir))
    parser.add_argument('--track-cluster', type=str, action='append', default=[],
                        help='track clusters scattering (--single-process mode only).')
    parser.add_argument('--print-track-cliques', action='store_const', const=True, default=False, 
                        help='print the options required to track cliques.')
    parser.add_argument('--add-edge', type=str, action='append', default=[],
                        help="explicitly add edge with the format 'rank rank' or 'rank rank weight' in the topology (only works for 'disconnected-cliques').")
    parser.add_argument('--topology-weights', type=str, default='metropolis-hasting', choices=['metropolis-hasting', 'equal-clique-probability'])
    parser.add_argument('--interclique-edges-per-node', type=int, default=1,
            help="number of interclique edges for each pair of cliques (only useful for 'minimal-node-symmetric-cliques' topology).")
    parser.add_argument('--learning-rate-schedule', type=float, nargs='+', default=[],
            help='Learning rate schedule list of EPOCH RATIO. (ex: 40 0.5 60 0.25) ')
    parser.add_argument('--batch-size-schedule', type=float, nargs='+', default=[],
            help='Batch size schedule list of EPOCH RATIO. (ex: 40 0.5 60 0.25) ')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
            help='Dataset for training and test. (default: mnist)')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'resnet20', 'linear-old', 'lenet', 'gn-lenet'],
            help='Model to learn. (default: linear)')
    parser.add_argument('--parallel-training', action='store_const', const=True, default=False, 
                        help='Train nodes in parallel using --nb-workers.')
    parser.add_argument('--nb-workers', type=int, default=8, metavar='N',
                        help='Number of parallel processes for training and logging models. Only applicable with \'--parallel-training\' (default: 8)')
    parser.add_argument('--accuracy-logging-interval', type=int, default=1, metavar='N',
                        help='Log validation and test accuracy every X epochs. (default: 1)')
    parser.add_argument('--validation-set-ratio', type=float, default=1., metavar='N',
                        help='Ratio of the validation set to use (total of 10,000 examples, default: 1.)')
    parser.add_argument('--clique-gradient', action='store_const', const=True, default=False, 
            help="Use the average gradient of the clique, instead of the local one. Only works with one of the clique topologies (ex: clique-ring, fully-connected-cliques, fractal-cliques ) ( default: False)")
    # "HACK": cliques is added as an argument to use the args object to pass options the assignment of nodes in cliques
    #         to experiments. We may want to make it configurable later but for now it should not be supplied externally.
    parser.add_argument('--cliques', type=list, default=[],
                        help='(internal) list of cliques (list of list of ranks)')
    parser.add_argument('--initial-averaging', action='store_const', const=True, default=False, 
            help="Average all models before training (only works with parallel-training and single-process) ( default: False)")
    # "HACK": averaging-neighbours is added as an argument to use the args object to pass options the assignment of nodes in cliques
    #         to experiments. We may want to make it configurable later but for now it should not be supplied externally.
    parser.add_argument('--averaging-neighbourhood', type=list, default=[],
                        help='(internal) list of neighbourhoods (list of sets of ranks)')
    parser.add_argument('--unbiased-gradient', action='store_const', const=True, default=False, 
            help="Use the average gradient of a subset of neighbours representing equally all classes. Only works with the 'greedy-diverse-10' topology ( default: False)")
    parser.add_argument('--remove-clique-edges', type=int, default=0, metavar='N', 
            help="Remove X random edges from each clique. Only works with clique-based topologies. ( default: 0)")
    
    args = parser.parse_args()

    if len(args.learning_rate_list) == 0:
        args.learning_rate_list.append(args.learning_rate)

    if len(args.batch_size_list) == 0:
        args.batch_size_list.append(args.batch_size)   

    if args.nodes_per_class is None:
        # By default, nodes per class should be balanced
        args.nodes_per_class = [ math.ceil( (args.nb_nodes / 10.) * args.local_classes ) for i in range(10) ] 
    elif args.nodes_per_class is not None and args.train_ratio is not None:
        print('Incompatible nodes-per-class and train-ratio options, please set only one.')
        sys.exit(1)
    elif args.nodes_per_class is not None and len(args.nodes_per_class) != 10:
        print('Invalid number of nodes per class, expected 10 integer values')
        sys.exit(1)
    elif sum(args.nodes_per_class) != args.nb_nodes * args.local_classes:
        print('Invalid number of nodes per class, should sum to nb-nodes * local-classes')
        sys.exit(1)
        
    if args.local_train_ratios is not None:
        print('Unsupported supplied local-train-ratios, will be set automatically based on other options')
        sys.exit(1)

    if len(args.global_train_ratios) != 10:
        print('Invalid global train ratios number, expected 10 float values')
        sys.exit(1)
    elif not all([ r >= 0. and r <= 1. for r in args.global_train_ratios ]):
        print('Invalid global train ratios {}, all values should be within [0.,1.] inclusive'.format(args.global_train_ratios))
        sys.exit(1)

    # train_ratio can now be None, in addition to a numerical value, to detect (above) cases
    # where it has explicitly been set by the user at the same time as nodes-per-class.
    # otherwise, it still uses its originally value of 1.
    if args.train_ratio is None and args.nodes_per_class is None:  
        args.train_ratio = 1.
        
    # train_ratio originally implicitly assumed that the numbers of nodes-per-class
    # were all equal. With the introduction of configurable nodes-per-class,
    # this is no longer the case. We set nodes-per-class here to be backward compatible
    # with the previous behaviour when it is not supplied by the user.
    if args.train_ratio is not None and args.nodes_per_class is None:
        # assume the number is balanced for all nodes
        args.nodes_per_class = [ 1./args.train_ratio for _ in range(10) ]
        assert all( x <= args.nb_nodes for x in args.nodes_per_class ), "Invalid train_ratio, some examples are repeated more than once"

    # Now derive the local_train_ratios from previous options
    if args.train_ratio is not None:
        args.local_train_ratios = [ r*args.train_ratio for r in args.global_train_ratios ]
    else:
        args.local_train_ratios = [ (1./n)*r for r,n in zip(args.global_train_ratios,args.nodes_per_class) ]

    # Ensure the class number is valid
    if args.first_node_class != None and \
       (args.first_node_class < 0 or\
        args.first_node_class > 9):
        print("Invalid first node class '{}', expected an int between 0 and 9".format(args.first_node_class))
        sys.exit(1)

    if len(args.first_node_position) > 0 and args.topology != 'grid':
        print("Unsupported first node position for non-grid topologies")
        sys.exit(1)

    if len(args.first_node_position) > 0 and args.topology == 'grid' and\
       len(args.first_node_position) != 2:
        print("Invalid first node position {} for a grid, expected 2 int values".format(args.first_node_position))
        sys.exit(1)

    if len(args.track_cluster) > 0:
        cluster_parser = argparse.ArgumentParser()
        cluster_parser.add_argument('cluster', metavar='N', type=int, nargs='+')
        args.track_cluster = [ cluster_parser.parse_args(c.split(' ')).cluster for c in args.track_cluster ] 

    if len(args.add_edge) > 0:
        if args.topology != 'disconnected-cliques':
            print("--add-edge only supported with a 'disconnected-cliques' topology")
            sys.exit(1)
        parsed = []
        for s in args.add_edge:
            s = s.split(' ')
            assert len(s) >= 2 and len(s) <= 3 and s[0].isdigit() and s[1].isdigit(), "Invalid edge {}, use 'rank rank' or 'rank rank weight' of type 'int int float'".format(s)
            w = None
            if len(s) == 3:
                try:
                    w = float(s[2])
                    assert w >= 0 and w < 1, "Invalid edge '{}', weight must be between [0,1["
                except:
                    print("Invalid edge '{}', weight must be a float")
                    sys.exit(1)
            if w is None:
                parsed.append([int(s[0]), int(s[1])])
            else:
                parsed.append([int(s[0]), int(s[1]), w])
        args.add_edge = parsed

    if len(args.learning_rate_schedule) % 2 != 0:
        print('learning rate schedule should be a list of EPOCH RATIO in pairs')
        sys.exit(1)
    else:
        schedule = args.learning_rate_schedule
        args.learning_rate_schedule = {
          int(schedule[i]): float(schedule[i+1]) for i in range(0,len(schedule),2)
        }
        args.learning_rate_schedule[0] = 1.0

    if len(args.batch_size_schedule) % 2 != 0:
        print('batch size schedule should be a list of EPOCH BATCH_SIZE in pairs')
        sys.exit(1)
    else:
        schedule = args.batch_size_schedule
        args.batch_size_schedule = {
          int(schedule[i]): int(schedule[i+1]) for i in range(0,len(schedule),2)
        }
        args.batch_size_schedule[0] = args.batch_size

    if args.clique_gradient and not (args.single_process or args.parallel_training):
        print('clique gradient only implemented for single-process or parallel-training modes')
        sys.exit(1)

    if args.unbiased_gradient and not (args.single_process or args.parallel_training):
        print('unbiased gradient only implemented for single-process or parallel-training modes')
        sys.exit(1)

    for bsz in args.batch_size_list:
        for lr in args.learning_rate_list:
            args.batch_size = bsz
            args.learning_rate = lr
            print(args)
            experiment(args)

