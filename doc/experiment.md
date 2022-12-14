# Experiment

Note: We only document here conventions for elements that have been useful for
at least one paper or otherwise repeated investigations. Current tool
implementations may still generate more files, events, or object properties that
are not documented here but could be left overs from previous one-off investigations.

## Conventions

### Single Experiment

An experiment is a meaningful unit of result(s) and will typically be used to
generate one or multiple curves for plots in papers.  We create one directory
per experiment, with a name specific to what the results are, and the following
sub-items:

````
EXPERIMENT/
  experiments.sh
  all/
    DATE-TIME1/
    DATE-TIME2/ 
    ...
````

````experiments.sh```` is a bash script that invokes the simulator tool(s) with
a specific set of input parameters to produce experiment results. Executing
````experiments.sh```` may invoke the simulator tool(s) one or multiple times:
each invocation generates a new sub-directory under ````all/```` with the date,
the time, and the machine on which the simulator was invoked (ex:
````2021-03-10-09:20:03-CET````). We call the combination of date, time,
input parameters and tool version a *run*. 

Naming runs with explicit DATE-TIME offers a number of benefits:
    1. Old results are never overwritten, only extended (see [Design
       Principles](./design.md)).
    2. Different subsets of parameter combinations can be run on different
       machines and merged later in the same directory with limited risks of
       conflicts (adding a -MACHINE suffix would eliminate it but has not been
       implemented yet).
    3. When and how the runs were obtained is explicitly referred by analysis
       tools that produce plots from the raw results of runs, simplifying
       traceability.

### Run (DATE-TIME)

Within one DATE-TIME run, the results are structured as follows:
````
DATE-TIME/
  events/
    0.json
    ...
    (N-1).json
    global.json 
  meta.json
  nodes.json
  topology.json
````

````events```` are lists of events generated by the nodes during simulation.
Each node has its own list, ````X.jsonlines```` where *X* is the index of the
node. In addition, ````global.jsonlines```` represents aggregated statistics on
all the nodes, that are periodically logged. `````.jsonlines```` files
serialize one event per line, in which each event is a JSON object. The content
of events is tool-specific.

````meta.json```` is a serialized JSON object that represents the input
parameters to the experiment, some additional information that may have been
derived from inputs, and the version number (i.e. latest git commit) of the
simulator: this is used for sanity checking and traceability.
````nodes.json```` is a serialized JSON object that describes the parameters of
each node in the topology, such as, for example, the classes of examples it has
as well as the number of examples it has for each class of examples: this is
used to validate and visualize the data partitioning.  ````topology.json```` is
a serialized JSON object that describes which edges exist between nodes and
what is their weight, if applicable: this is used to analyze and visualize the
topology.

### Set of Experiments

The set of all experiment results are under 'results' within the repository of a paper.

````
paper/
  main.tex
  main.bib
  ...
  results/
    EXPERIMENT-1/
      experiments.sh
    EXPERIMENT-2/
      experiments.sh
    ...
````

Because experiments are deterministic and can be fully replicated only from the
````experiments.sh```` script, only that script is committed for version
control. We do keep a copy of all experiment results used for a paper with
automatic backup outside of version control (ex: with Time Machine on MacOS).

## Input: JSON Parameters

The are many input parameters currently used to configure an experiment. We
list and explain the most important. For clarity, they are presented
in categories in different objects. The current implementation merges them 
all in a single object.

For all options listed below, a command-line option is invoked with
````--option-name [VALUE]```` and will be automatically converted to
````"option_name": VALUE```` in the JSON object representing parameters (note
the ````--```` prefix as well as the conversion of ````-```` to  ````_````).

### Topology

The following options determine the topology, i.e. how many nodes are connected,
how they are connected, and how to assign weights to edges of nodes:
````
{
 'nb_nodes': 1,
 'topology': 'fully_connected',
 'metric': 'dissimilarity',
 'topology_weights': 'metropolis-hasting'
}
````

````nb_nodes```` defines how many nodes to simulate;
````topology```` defines how many edges to add and between which nodes, according to
                 different topology generation algorithms;
````metric```` selects how to choose neighours, i.e. 'dissimilarity' choses
neighors that have examples of different classes, 'similarity' choses neighbors
that have examples of the same class(es), and 'random' does not take into
account the classes present in a node;
````topology_weights```` defines which algorithm to use to assign weights to edges between nodes.

### Partitioning

The following options determine how the dataset will be split between nodes:
````
{
 'local_classes': 10,
 'nodes_per_class': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
````

````local_classes```` defines how many classes should be present in the local distribution of every node;
````nodes_per_class```` defines how many nodes should have examples of each class.


### Training Algorithm

The following options determine which training algorithm to use, with corresponding 
hyper-parameters:
````
{
 'dist_optimization': 'd-psgd',
 'batch_size': 12800,
 'learning_rate': 0.1,
 'learning_momentum': 0.0
}
````
````dist_optimization```` defines the algorithm to use. Most of our studies
simply use 'd-psgd' (aka D-SGD) for now.
````batch_size```` defines the maximum number of samples used to perform one
gradient step. Samples in mini-batches are randomly chosen without replacement
until the local distribution has been fully traversed. The last mini-batch may
have less than the number given, if there are not enough samples left. In the
mini-batch following the last, sampling is re-started from the entire local
distribution of a node.
````learning_rate```` , or (gradient) step size, defines a ratio that is
multiplied with the gradients computed during training, effectively modulating 
how much the model weights can for each gradient step.
````learning_momentum````, defines the ratio for momentum, which accumulates
the previous gradients computed and amplifies the step size in the direction
of the (approximate) average gradient of the last steps.

### Dataset

The following options determine the dataset and number of examples to use for training.
````
{
 'dataset': 'mnist',
 'global_train_ratios': [0.802568, 0.802568, 0.802568, 0.802568, 0.802568, 
                         0.802568, 0.802568, 0.802568, 0.802568, 0.802568]
}
````

````dataset```` defines the data to train on, currently only 'mnist' and
'cifar10' are supported.
````global_train_ratios```` defines the ratio of examples of each class to
distribute between nodes. For datasets like MNIST, in which some classes have
more examples than others, the smaller classes are first randomly resampled to
obtain an equal number of examples as the largest class, then the ratio is
applied. In the example above, using a global ratio of 0.802568 with 'mnist'
actually corresponds to taking no more examples than the size of the smallest
class.

### Model

The following option determines the kind of machine learning model to optimize
during training:
````
{
 'model': 'linear'
}
````

````model```` defines the model to use: 'linear' is a linear regression model,
and 'gn-lenet' is a deep convolutional network with group normalization.

### Simulation

The following options determine how the simulator behaves:
````
{
 'nb_epochs': 100,
 'seed': 1,
 'single_process': True,
 'nb_logging_processes': 8,
 'accuracy_logging_interval': 1,
}
````

````nb_epochs```` defines how many epochs, i.e. complete pass over the entire
dataset, to run before stopping.
````seed```` sets the initial seed of the pseudo-random number generator for
all operations that require sampling and random choices.
````single_process```` controls whether all the nodes should be simulated
within a single process (True) or with OS processes. In general the former is
faster.
````nb_logging_processes```` defines how many parallel processes to use to
evaluate the accuracy of models during training. This step is usually more time
consumming than proper training.
````accuracy_logging_interval```` defines how often the accuracy of models
should be measured, in epochs. ````1```` measures every epoch, and for all
other values ````X```` greater than 1, accuracy is measured every X epochs.


### Automatic Parameters

The following options are gathered automatically when the simulator is invoked,
for traceability of results:
````
{
 'script': 'simulate.py',
 'git-hash': 'a786ec1'
}
````

````script```` is the name of the script that was invoked.
````git-hash```` is the hash of the latest recorded git commit of the git
repository from which the simulator is run. This serves as automatic "version"
tracking. Note that uncommitted changes to the simulator won't have a distinct
git-hash, so it is preferable to always commit changes before running important
experiments.

## Output: JSON Events

We are logging two types of events: local to individual nodes and global for
the entire population of nodes.

### Local Events: events/X.jsonlines

Local events report on the status of individual nodes during training: each
node generates a sequence of events in a dedicated file
````events/X.jsonlines````, where X is the node's rank, i.e. its numerical
unique identifier. We log local events for later post-processing to compare  
the behavior of different variations of input parameters, typically to derive
curves to compare the speed of convergence both min, max as well as average 
over all nodes.

All events share the following properties:
````
{
  "type": "accuracy", 
  "data": "train" || "valid" || "test",
  "rank": 0, 
  "epoch": 1, 
  "loss": 0.8554201722145081, 
  "accuracy": 0.7611555555555556
}
````

````type```` denotes the type of event that happened. For now, nodes only
generate *accuracy* event, i.e. an event that reports what is the accuracy of
the model of node *rank* after training *epochs* (which may implicitly contain
multiple mini-batches that are not evaluated).
````data```` is the partition of the dataset that generated an event: "train"
means the training dataset that is local to the node; "valid" means a partition
of the original dataset that is not used for training but only to select
hyper-parameters; "test" means the partition of the dataset that is neither
used to train nor to select hyper-parameters, only to report final results.
````loss```` is the average loss computed on a complete subset of the dataset:
for "train", this is the loss over the full local data partition; for "valid"
and "test", this is the loss over the global validation/test datasets that all
nodes share. 
````accuracy```` is the ratio of train/validation/test examples that were
correctly classified.

#### Additional properties for Training Events

In addition to the above, training events also have the following property:
````
{
  "running_loss": 1.4533979892730713, 
}
````

````running_loss````: contrary to "loss" on "train" data, that is computed on a
node's entire local dataset at a single point in time, "running_loss" is the
average loss over all mini-batches that sequentially happened in the last
epoch. Both "loss" and "running_loss" are computed over the same dataset but
"running_loss" is over changing model parameters between different mini-batch.

### Global Events: events/global.jsonlines

Global events report on the aggregate status of all nodes. Strictly speaking,
the same information could be derived from local events with relevant and
sufficient details. However, in practice, some interesting metrics require an impractical
(several GBs) amount of data in local events to be computed after training. For 
those cases, we log only the aggregate information obtained from the entire population
of nodes during training.

At the moment, the only global event we log is the [consensus
distance](https://arxiv.org/abs/2102.04828), which we had implemented and
called "model-scattering" prior to the consensus distance publication, with
the following properties:
````
{
  "type": "model-scattering",
  "epoch": Number,
  "distance_to_center": {
    "center": [ Number, ... ],
    "avg": Number
  }
}
````

````epoch```` is the number of epochs of training that were completed prior to
computing the metric.  ````distance_to_center```` is the consensus distance.
````distance_to_center:center```` is the average of all node models at that
point in time.  ````distance_to_center:avg```` is the average of the Euclidian
distance between all nodes' models and the ````distance_to_center:center````.


