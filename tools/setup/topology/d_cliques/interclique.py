import math
from random import Random

def ring(_cliques, _edges, params):
    # Create a copy to modify them without affecting the args
    cliques = [ set(c) for c in _cliques ]
    edges = { rank: set(_edges[rank]) for rank in _edges }

    # create the ring
    prev =  cliques[-1].pop() if len(cliques[-1]) > 1 else list(cliques[-1])[0]
    for clique in cliques:
        current = clique.pop() if len(cliques[-1]) > 1 else list(cliques[-1])[0]
        # add edge between current and prev
        edges[prev].add(current)
        edges[current].add(prev)
        # prepare for next
        prev = clique.pop()
    return edges

def fractal(_cliques, _edges, params):
    edges = { rank: set(_edges[rank]) for rank in _edges }
    cliques = [ { m:0 for m in c } for c in _cliques ]

    clique_size = params['dataset']['nb-classes']
    seed = params['meta']['seed']
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
        cliques = [ connect(toconnect[i:i+clique_size]) for i in range(0,len(toconnect),clique_size) ]

    return edges

def fully_connected(_cliques, _edges, params):
    edges = { rank: set(_edges[rank]) for rank in _edges }
    cliques = [ { m:0 for m in c } for c in _cliques ]

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

# Add additional edges per node in a "smallworld" topology built
# upon a ring: cliques preferentially attach to cliques closer on
# the ring (clock-wise and counter-clockwise) with exponentially less edges
# the further away on the ring.
def smallworld(_cliques, _edges, params):
    edges = { rank: set(_edges[rank]) for rank in _edges }
    cliques = [ { m:0 for m in c } for c in _cliques ]

    assert 'meta' in params.keys(), "Missing 'meta' params"
    seed = params['meta']['seed']
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

def get(name):
    if name == 'ring': return ring
    elif name == 'fractal': return fractal
    elif name == 'fully-connected': return fully_connected
    elif name == 'smallworld': return smallworld
    else:
        raise Exception("Invalid interconnect name '{}'".format(name))
