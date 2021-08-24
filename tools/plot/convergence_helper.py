import os
import json
import scipy.optimize
import numpy as np
import math
import sys

def pred_epoch(x, y):
    sqrt = lambda x, a, b, c: a*x**(1./b) + c
    pred = lambda y, a, b, c: ((y-c)/a)**b
    try:
        params = scipy.optimize.curve_fit(sqrt, x[1:], y[1:])
    except RuntimeError:
        params = [[0.0001,0.0001,0.0001]]
    except TypeError:
        params = [[0.0001,0.0001,0.0001]]
    return {
        0.80: pred(0.80, params[0][0], params[0][1], params[0][2]),
        0.85: pred(0.85, params[0][0], params[0][1], params[0][2]),
        0.90: pred(0.90, params[0][0], params[0][1], params[0][2]),
        0.91: pred(0.91, params[0][0], params[0][1], params[0][2]),
        0.92: pred(0.92, params[0][0], params[0][1], params[0][2])
    }

def training_loss_from_dir(entry):
    with open(os.path.join(entry, 'meta.json'), 'r') as meta_file:
        meta = json.load(meta_file)

    events = {}
    losses = {}
    events_dir = os.path.join(entry, 'events')
    for _, _, files in os.walk(events_dir):
        for log_path in files:
            rank_str = log_path[:log_path.find('.jsonlines')]
            if rank_str == 'global':
                continue
            try:
                rank = int(rank_str)
            except e:
                print(e)
                print(rank_str)
                print(entry)
                os.exit(1)

            with open(os.path.join(events_dir, log_path), 'r') as log_file:
                events[rank] = [ json.loads(line) for line in log_file ]
                
            losses[rank] = { e['epoch']:e for e in events[rank] if e['type'] == 'accuracy' and e['data'] == 'train'}

    convergence = {
      'min': [],
      'max': [],
      'avg': [],
      'delta-avg': [],
      'std': [],
      'nb': [],
      'sampling_epochs': [ epoch for epoch in losses[0] ]
    }

    for epoch in range(1, meta['nb_epochs'] + 1):
        loss = [ losses[rank][epoch]['loss'] for rank in range(0, meta['nb_nodes']) 
                if rank in losses and len(losses[rank]) >= epoch ]
        if len(loss) == 0:
            break
        convergence['min'].append(min(loss))
        convergence['max'].append(max(loss))
        avg = sum(loss)/len(loss)
        convergence['avg'].append(avg)
        convergence['std'].append(sum([ abs(avg-loss[i]) for i in range(0,len(loss))])/len(loss))
        convergence['nb'].append(len(loss))

    if len(convergence['avg']) < 2:
        print('error insufficient number of measurements for {}'.format(entry))
        sys.exit(1)

    return convergence

def running_training_loss_from_dir(entry):
    with open(os.path.join(entry, 'params.json'), 'r') as params_file:
        params = json.load(params_file)

    events = {}
    losses = {}
    events_dir = os.path.join(entry, 'events')
    for _, _, files in os.walk(events_dir):
        for log_path in files:
            rank_str = log_path[:log_path.find('.jsonlines')]
            if rank_str == 'global':
                continue
            try:
                rank = int(rank_str)
            except Exception as e:
                print(e)
                print(rank_str)
                print(entry)
                raise e

            with open(os.path.join(events_dir, log_path), 'r') as log_file:
                events[rank] = [ json.loads(line) for line in log_file ]
                
            losses[rank] = { e['epoch']:e for e in events[rank] if e['type'] == 'accuracy' and e['data'] == 'train'}

    convergence = {
      'min': [],
      'max': [],
      'avg': [],
      'delta-avg': [],
      'std': [],
      'nb': [],
      'sampling_epochs': [ epoch for epoch in losses[0] ]
    }

    for epoch in range(1, params['simulator']['nb-epochs'] + 1):
        loss = [ losses[rank][epoch]['running_loss'] for rank in range(0, params['nodes']['nb-nodes']) 
                if rank in losses and len(losses[rank]) >= epoch ]
        if len(loss) == 0:
            break
        convergence['min'].append(min(loss))
        convergence['max'].append(max(loss))
        avg = sum(loss)/len(loss)
        convergence['avg'].append(avg)
        convergence['std'].append(sum([ abs(avg-loss[i]) for i in range(0,len(loss))])/len(loss))
        convergence['nb'].append(len(loss))

    if len(convergence['avg']) < 2:
        print('error insufficient number of measurements for {}'.format(entry))
        sys.exit(1)

    return convergence

def training_accuracy(entry):
    with open(os.path.join(entry, 'params.json'), 'r') as params_file:
        params = json.load(params_file)

    events = {}
    acc = {}
    events_dir = os.path.join(entry, 'events')
    for _, _, files in os.walk(events_dir):
        for log_path in files:
            rank_str = log_path[:log_path.find('.jsonlines')]
            if rank_str == 'global':
                continue
            try:
                rank = int(rank_str)
            except e:
                print(e)
                print(rank_str)
                print(entry)
                os.exit(1)

            with open(os.path.join(events_dir, log_path), 'r') as log_file:
                events[rank] = [ json.loads(line) for line in log_file ]
                
            acc[rank] = { e['epoch']:e for e in events[rank] if e['type'] == 'accuracy' and e['data'] == 'train'}

    convergence = {
      'min': [],
      'max': [],
      'avg': [],
      'delta-avg': [],
      'std': [],
      'nb': [],
      'sampling_epochs': [ epoch for epoch in acc[0] ]
    }

    for epoch in range(1, params['simulator']['nb-epochs'] + 1):
        loss = [ acc[rank][epoch]['accuracy'] for rank in range(0, params['nodes']['nb-nodes']) 
                if rank in acc and len(acc[rank]) >= epoch ]
        if len(loss) == 0:
            break
        convergence['min'].append(min(loss))
        convergence['max'].append(max(loss))
        avg = sum(loss)/len(loss)
        convergence['avg'].append(avg)
        convergence['std'].append(sum([ abs(avg-loss[i]) for i in range(0,len(loss))])/len(loss))
        convergence['nb'].append(len(loss))

    if len(convergence['avg']) < 2:
        print('error insufficient number of measurements for {}'.format(entry))
        sys.exit(1)

    return convergence

def from_dir(entry, test_set='test'):
    with open(os.path.join(entry, 'params.json'), 'r') as params_file:
        params = json.load(params_file)
    nb_epochs = params['simulator']['nb-epochs']

    events = {}
    accuracies = {}
    events_dir = os.path.join(entry, 'events')
    for _, _, files in os.walk(events_dir):
        for log_path in files:
            if log_path == 'global.jsonlines':
                continue
            rank_str = log_path[:log_path.find('.jsonlines')]
            try:
                rank = int(rank_str)
            except Exception as e:
                print(e)
                print(rank_str)
                print(entry)
                sys.exit(1)

            with open(os.path.join(events_dir, log_path), 'r') as log_file:
                events[rank] = [ json.loads(line) for line in log_file ]
                
            accuracies[rank] = { e['epoch']:e for e in events[rank] if e['type'] == 'accuracy' and e['data'] == test_set }

    convergence = {
      'min': [],
      'max': [],
      'avg': [],
      'delta-avg': [],
      'std': [],
      'nb': [],
      'epochs': {},
      'classes': {},
      'sampling_epochs': [ epoch for epoch in accuracies[0] ]
    }

    for i in range(10):
        convergence['classes'][i] = {
            'min': [],
            'max': [],
            'avg': [],
            'delta-avg': [],
            'std': [],
            'nb': []
        }

    def class_accuracy(confusion, target):
        total = sum([confusion[pred][target] for pred in range(10)])
        return float(confusion[target][target]) / total

    for epoch in range(0, nb_epochs + 1):
        if not epoch in accuracies[rank]:
            continue

        acc = [ accuracies[rank][epoch]['accuracy'] for rank in range(0, params['nodes']['nb-nodes']) 
                if rank in accuracies ]
        if len(acc) == 0:
            break
        convergence['min'].append(min(acc))
        convergence['max'].append(max(acc))
        avg = sum(acc)/len(acc)
        convergence['delta-avg'].append(avg - convergence['avg'][-1] if len(convergence['avg']) > 0 else 0.)
        convergence['avg'].append(avg)
        convergence['std'].append(sum([ abs(avg-acc[i]) for i in range(0,len(acc))])/len(acc))
        convergence['nb'].append(len(acc))

    if len(convergence['avg']) > 2:
        preds = pred_epoch(range(0, len(convergence['avg'])), convergence['avg'])
    else:
        print('error insufficient number of measurements for {}'.format(entry))
        sys.exit(1)

    if len(convergence['avg']) == nb_epochs:
        for acc in [0.80, 0.85, 0.90, 0.91, 0.92]:
            for e in range(0, nb_epochs+1):
                if convergence['avg'][e] >= acc:
                    convergence['epochs'][acc] = e
                    break
            if acc not in convergence['epochs']:
                if math.isnan(preds[acc]) or preds[acc] == math.inf or math.ceil(preds[acc]) > 999.:
                    convergence['epochs'][acc] = None
                else:
                    pred = int(math.ceil(preds[acc]))
                    convergence['epochs'][acc] = -(pred + 1) if pred == nb_epochs else -pred

    neg_delta_avg = [ d for d in convergence['delta-avg'] if d < 0. ]
    convergence['avg-neg-delta-avg'] = sum(neg_delta_avg) / len(neg_delta_avg) if len(neg_delta_avg) > 0  else 0

    return convergence
