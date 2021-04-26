See https://gitlab.epfl.ch/sacs/distributed-ml/d-cliques/-/tree/master/results
for more experiment examples. When running from this repository, the $TOOLS
variable should be updated to point to ````../tools````, and ````sgp-mnist.py```` should be
replaced with ````simulate.py````. Each experiment will be logged into an all/$DATETIME directory.

Multiple experiment results can be used to generate graphs using
````tool/plot\_convergence.py````.  See
https://gitlab.epfl.ch/sacs/distributed-ml/d-cliques/-/blob/master/main.tex for
examples (you still need to update paths and experiment names for your
setting).
