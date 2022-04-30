#!/usr/bin/env python
import json
import os
import pickle
import sys
from importlib import import_module

import torch
import torch.nn.functional as F

import setup.dataset as d
import setup.meta as m


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    r = []
    for i in range(0, n):
        r.append(l[i::n])
    return r


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


if __name__ == "__main__":
    rundir = sys.argv[1]
    total_loggers = int(sys.argv[2])
    my_logger_index = int(sys.argv[3])
    params = m.params(rundir)

    test_set = torch.utils.data.DataLoader(d.test(params), 100)
    validation_set = torch.utils.data.DataLoader(d.valid(params), 100)

    # Read the directory with models and assign them to this logger instance
    models_dir = os.path.join(rundir, "models")
    meta_dir = os.path.join(rundir, "meta")
    filenames = os.listdir(models_dir)
    assignments = chunks(filenames, total_loggers)
    my_models = assignments[my_logger_index]
    print("My assigned models: %s" % my_models)

    # Start computing accuracies
    for filename in my_models:
        model = import_module(params['model']['module']).create(params)
        with open(os.path.join(models_dir, filename), "rb") as in_file:
            serialized_model = in_file.read()
            model.load_state_dict(pickle.loads(serialized_model))

        node = None
        with open(os.path.join(meta_dir, filename), "r") as meta_file:
            node = json.load(meta_file)

        for type, dataset in [("test", test_set), ("valid", validation_set)]:
            print("Computing step %d accuracy for node %d" % (node["step"], node["rank"]))
            accuracy, test_loss = model_accuracy(model, dataset, params)
            event_file = os.path.join(rundir, "events", "{}.jsonlines".format(node["rank"]))
            with open(event_file, 'a') as events:
                events.write(json.dumps({
                    "type": "accuracy",
                    "data": type,
                    "rank": node["rank"],
                    "epoch": node["epoch"],
                    "step": node["step"],
                    "loss": test_loss,
                    "accuracy": accuracy,
                    "timestamp": m.now()
                }) + '\n')
