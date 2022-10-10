#!/usr/bin/env python
import argparse
import logging
import json
import pickle
import copy
import os
import math
import subprocess
from importlib import import_module
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.multiprocessing import Process, JoinableQueue, Pipe
import setup.meta as m
import setup.dataset as d
import setup.model
import statistics
from setup import get_das5_nodes


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

    def state(self, nodes, state):
        if not nodes:
            return

        logging.info('logger.state step %d (%d nodes)', state["step"], len(nodes))
        if not self.params['logger']['skip-training']:
            for node in nodes:
                self.log_train_accuracy(node, state)

        # Serialize the average of all models
        global_model_path = None
        global_meta_path = None
        if self.params['logger']['log-global-model-accuracy']:
            epoch = nodes[0]["epoch"]
            assert all(map(lambda n: n["epoch"] == epoch, nodes)), "Inconsistent epochs between nodes"
            center = setup.model.average([ n['model'] for n in nodes ])

            global_dir = os.path.join(self.rundir, "global")
            if not os.path.exists(global_dir):
                os.mkdir(global_dir)

            global_model_dir = os.path.join(self.rundir, "global", "model")
            if not os.path.exists(global_model_dir):
                os.mkdir(global_model_dir)

            global_meta_dir = os.path.join(self.rundir, "global", "meta")
            if not os.path.exists(global_meta_dir):
                os.mkdir(global_meta_dir)

            global_model_path = os.path.join(self.rundir, "global", "model", str(state["step"]))
            global_meta_path = os.path.join(self.rundir, "global", "meta", str(state["step"]))
            with open(global_model_path, "wb") as center_file:
                center_file.write(pickle.dumps(center.state_dict()))
            with open(global_meta_path, "w") as center_meta_file:
                json.dump({
                    "step": state["step"],
                    "epoch": epoch
                }, center_meta_file)

        # Serialize all the individual models that should be checked
        model_paths = []
        for node in nodes:
            serialized_model = pickle.dumps(node["model"].state_dict())
            filename = "%d_%d" % (state["step"], node["rank"])
            models_dir = os.path.join(self.rundir, "models")
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)

            model_path = os.path.join(models_dir, filename)
            model_paths.append(model_path)
            with open(model_path, "wb") as model_file:
                model_file.write(serialized_model)

        # Serialize meta information about nodes (ex: epoch)
        meta_paths = []
        meta_dir = os.path.join(self.rundir, "meta")
        if not os.path.exists(meta_dir):
            os.mkdir(meta_dir)
        for node in nodes:
            filename = "%d_%d" % (state["step"], node["rank"])
            meta_path = os.path.join(meta_dir, filename)
            meta_paths.append(meta_path)
            with open(meta_path, "w") as meta_file:
                json.dump({
                    "rank": node["rank"],
                    "step": state["step"],
                    "epoch": node["epoch"]
                }, meta_file)

        # Depending on the number of reserved nodes, start several logging instances
        reserved_das5_nodes = get_das5_nodes(os.environ["SLURM_JOB_NODELIST"])
        total_loggers = len(reserved_das5_nodes) * 3  # Three instances per reserved DAS5 node

        procs = []
        logger_ind = 0
        for reserved_das5_node in reserved_das5_nodes:
            for ind in range(3):
                cmd = "ssh node%d \"source /home/spandey/venv3/bin/activate && PYTHONPATH=%s python %s/simulate/run_logger.py %s %d %d %d\"" % (reserved_das5_node, os.getcwd(), os.getcwd(), self.rundir, total_loggers, logger_ind, state["step"])
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                procs.append(p)
                logger_ind += 1

        for p in procs:
            p.wait()
            logging.info("logger.state process finished")
            #output, err = p.communicate()

        # Remove the temporary files
        if global_model_path:
            os.unlink(global_model_path)
        if global_meta_path:
            os.unlink(global_meta_path)

        for model_path in model_paths:
            os.unlink(model_path)
        for meta_path in meta_paths:
            os.unlink(meta_path)

        #self.log_test_accuracy(node, state)

    def loss(self, losses):
        for node_rank, loss in losses.items():
            self.running_loss[node_rank] += loss
            self.running_loss_count[node_rank] += 1

    def log_train_accuracy(self, node, state):
        params = self.params
        rank = node['rank']
        epoch = node['epoch']
        logging.info('logger.log_train_accuracy node {}'.format(rank))
        model = node['model']
        model.eval()
        event_file = node['event-file']
        total_loss = 0.0
        num_batches = 0.0
        correct = 0.0
        example_number = 0.0
        train = torch.utils.data.DataLoader(node['train-set'], 1000)

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

    def log_test_accuracy(self, node, state):
        event_file = node['event-file']
        step = state['step']
        self.tasks.put((node['rank'], node['epoch'], step, pickle.dumps(node['model'].state_dict()), event_file))
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
    parser.add_argument('--accuracy-logging-interval-steps', type=int, default=0, metavar='S',
                        help='Log validation and test accuracy every X steps. (default: 0)')
    parser.add_argument('--skip-validation', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on validation set. ( default: False)")
    parser.add_argument('--skip-testing', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on test set. ( default: False)")
    parser.add_argument('--skip-training', action='store_const', const=True, default=False, 
            help="Skip accuracy measurements on training set. ( default: False)")
    parser.add_argument('--log-consensus-distance', action='store_const', const=True, default=False,
            help="Whether to periodically log the consensus distance. ( default: False)")
    parser.add_argument('--log-global-model-accuracy', action='store_const', const=True, default=False,
            help="Whether to periodically log the training, validation, and test accuracy " +
                 "of the global model. ( default: False)")

    args = parser.parse_args()
    rundir = m.rundir(args)

    logger = {
        'nb-processes': args.nb_processes,
        'accuracy-logging-interval': args.accuracy_logging_interval,
        'accuracy-logging-interval-steps': args.accuracy_logging_interval_steps,
        'skip-testing': args.skip_testing,
        'skip-validation': args.skip_validation,
        'skip-training': args.skip_training,
        'log-consensus-distance': args.log_consensus_distance,
        'log-global-model-accuracy': args.log_global_model_accuracy
    }
    m.extend(rundir, 'logger', logger) # Add to run parameters

    if args.rundir is None:
        print(rundir)
