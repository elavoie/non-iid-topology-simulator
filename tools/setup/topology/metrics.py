from random import Random

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
