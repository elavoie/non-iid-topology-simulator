def skew(d1, d2):
    def valid(d):
        return all(map(lambda x: x >= 0 and x <= 1, d)) and\
               sum(d) >= 0.999999 and\
               sum(d) <= 1.000001

    assert len(d1) == len(d2), "Inconsistent length between {} and {}".format(d1,d2)
    assert valid(d1), "Invalid d1 ({}), all values should be between 0 and 1, and sum to 1"
    assert valid(d2), "Invalid d2 ({}), all values should be between 0 and 1, and sum to 1"

    return sum([ abs(x-y) for x,y in zip(d1,d2) ])

def dist(nodes):
    nbs = [ 0 for _ in nodes[0]['samples'] ]
    for n in nodes:
        samples = n['samples']
        for i in range(len(nbs)):
            r = samples[i]
            nbs[i] += (r[1]-r[0])
    total = sum(nbs)
    return [ float(x)/total for x in nbs ]

