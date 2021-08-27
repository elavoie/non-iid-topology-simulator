# Decentralized Learning Simulator

## Installing dependencies on ````labostrexXYZ```` machines

### Install Conda (Python Package Manager)

See https://docs.conda.io/en/latest/miniconda.html#linux-installers. Successfully tested with Python 3.7:

Download and make miniconda script executable.
````
$ wget Miniconda3-py...-Linux-x86_64.sh ./
$ chmod +x Miniconda3-py...-Linux-x86_64.sh 
````

Run the installer:
````
$ ./Miniconda3-py...-Linux-x86_64.sh 
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
````

Activate the environment in the current shell:
````
$ source ~/.bashrc
````

### Install Pytorch Packages

````
$ conda install pytorch torchvision
````

Successfully tested with pytorch v1.7.1, torchvision 0.8.2.

### Install Git

````
$ sudo apt-get install git
````
 
### Setup Commit Info
````
$ git config --global user.email "email@mail.com"
$ git config --global user.name "First Last"
````

### Install the Simulator

````
$ git clone https://gitlab.epfl.ch/sacs/distributed-ml/non-iid-topology-simulator.git
````

### Run the test experiment

````
$ cd non-iid-topology-simulator
$ ./test.sh
````

## More Usage Examples

See https://gitlab.epfl.ch/sacs/distributed-ml/d-cliques/-/tree/master/results-v2
for more experiment examples: each experiment is named 'experiments.sh' in a
sub-directory.
