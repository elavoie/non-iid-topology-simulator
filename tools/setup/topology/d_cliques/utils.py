from random import Random

def remove_clique_edges(_edges, _cliques, params):
    seed = params['meta']['seed']
    rand = Random()
    rand.seed(seed)

    edges = { rank: set(_edges[rank]) for rank in _edges }
    cliques = [ list(clique) for clique in _cliques ] 

    for clique in cliques:
        candidates = [] 
        for i in range(len(clique)-1):
            for j in range(i+1, len(clique)):
                candidates.append((clique[i],clique[j]))
        rand.shuffle(candidates)
        for (m,n) in candidates[:params['topology']['remove-clique-edges']]:
            edges[m].remove(n)
            edges[n].remove(m)

    return edges, cliques
