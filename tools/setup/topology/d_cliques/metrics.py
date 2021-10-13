from math import log, sqrt
def valid(d):
    """Checks if the density given as an array or list sums up to 1.

    Args:
        d (List): Density as a list of individual probabilities.

    Returns:
        (Bool): Either or not the probability sums up to 1.
    """

    return all(map(lambda x: x >= 0 and x <= 1, d)) and\
            sum(d) >= 0.999999 and\
            sum(d) <= 1.000001

def is_density(d1,d2):
    """Checks different criteria and assumption before comparing two densities.

    Args:
        d1 (List): Density
        d2 (List): Density
    """
    assert len(d1) == len(d2), "Inconsistent length between {} and {}".format(d1,d2)
    assert valid(d1), "Invalid d1 ({}), all values should be between 0 and 1, and sum to 1"
    assert valid(d2), "Invalid d2 ({}), all values should be between 0 and 1, and sum to 1"

def skew(d1, d2):
    is_density(d1,d2)

    return sum([ abs(x-y) for x,y in zip(d1,d2) ])

def relative_entropy(d1, d2):
    is_density(d1,d2)

    return sum([ x*log(x/y) for x,y in zip(d1,d2) ])

def symmetric_relative_entropy(d1, d2):
    is_density(d1,d2)

    return sum([ 0.5*x*log(x/y)+0.5*y*log(y/x) for x,y in zip(d1,d2) ])

def hellinger(d1, d2):
    is_density(d1,d2)

    return sqrt(sum([ (sqrt(x)-sqrt(y))**2 for x,y in zip(d1,d2) ]))

def euclidean(d1, d2):
    is_density(d1,d2)

    return sqrt(sum([ (x-y)**2 for x,y in zip(d1,d2) ]))

def dist(nodes):
    nbs = [ 0 for _ in nodes[0]['samples'] ]
    for n in nodes:
        samples = n['samples']
        for i in range(len(nbs)):
            r = samples[i]
            nbs[i] += (r[1]-r[0])
    total = sum(nbs)
    return [ float(x)/total for x in nbs ]

