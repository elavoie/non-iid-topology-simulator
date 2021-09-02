import torch

def compute_weights(nodes, edges, params):
    weights = torch.zeros((len(nodes), len(nodes)))
    if params['weights'] == 'equal-clique-probability':
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue # Value derived from other values
                elif j in edges[i]:
                    weights[i,j] = edges[i][j]

        for i in range(len(nodes)):
            weights[i,i] = 1. - weights[i,:].sum()
    elif params['weights'] == 'metropolis-hasting':
        # using Metropolis Hasting (http://www.web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf, eq 4)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue # Do it last
                elif j in edges[i]:
                    weights[i,j] = 1./(max(len(edges[i]), len(edges[j])) + 1)

        for i in range(len(nodes)):
            weights[i,i] = 1. - weights[i,:].sum()
    else:
        raise Exception("Unimplemented topology_weights {}".format(args.topology_weights))
    eps = torch.finfo(torch.float32).eps
    assert all((weights.sum(axis=0) - 1.0) < 10*eps), "Weights should sum to 1. Sum is off by {}".format(weights.sum(axis=0) - 1.0)
    assert all((weights.sum(axis=1) - 1.0) < 10*eps), "Weights should sum to 1. Sum is off by {}".format(weights.sum(axis=1) - 1.0)

    return weights.tolist()

