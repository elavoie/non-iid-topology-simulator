import copy
import time

import numpy as np
import matplotlib.pyplot as plt


class Clique:
    def __init__(self, node_distribution):
        self.all_nodes = [np.array(copy.deepcopy(node_distribution))]
        self.clique_distribution = self.all_nodes[-1]
        self.skew_over_added_nodes = []

    def update_clique_distribution(self):
        self.clique_distribution += self.all_nodes[-1]

    def add_node(self, node_distribution):
        self.all_nodes.append(node_distribution)
        self.update_clique_distribution()
        return self

    def get_clique_distribution(self):
        return self.clique_distribution


def prob(clique):
    clique_distribution = clique.get_clique_distribution()
    return clique_distribution / clique_distribution.sum()


def skew(clique, global_distribution):
    prob_clique = prob(clique)
    prob_global_distribution = global_distribution / global_distribution.sum()
    probabilities_difference = np.abs(prob_clique - prob_global_distribution)
    return probabilities_difference.sum()


def create_nodes(random_generator, n_classes, n_nodes):
    return random_generator.uniform(100, 10001, (n_nodes, n_classes))


def one_iteration_of_alg4(distributed_cliques, global_distribution,
                          node_distribution, average_skew_iterations=None,
                          max_skew_iterations=None, min_skew_iterations=None,
                          return_index_not_add=False, max_n_nodes=None):
    if distributed_cliques:
        lowest_skew = 100
        index_of_lowest_skew_between_cliques = -1
        for i in range(len(distributed_cliques)):
            clique = distributed_cliques[i]
            if max_n_nodes is None or len(clique.all_nodes) < max_n_nodes:
                temporary_clique = copy.deepcopy(clique).add_node(
                    node_distribution)
                temporary_clique_skew = skew(temporary_clique,
                                             global_distribution)
                clique_skew = skew(clique, global_distribution)
                curr_diff = temporary_clique_skew - clique_skew
                if index_of_lowest_skew_between_cliques < 0 or curr_diff < lowest_skew:
                    index_of_lowest_skew_between_cliques = i
                    lowest_skew = curr_diff
                del temporary_clique

        if return_index_not_add:
            return index_of_lowest_skew_between_cliques, lowest_skew

        if lowest_skew >= 0:
            distributed_cliques.append(Clique(node_distribution))
        else:
            distributed_cliques[index_of_lowest_skew_between_cliques].add_node(
                node_distribution)
    else:
        distributed_cliques.append(Clique(node_distribution))
    average_skew, max_skew, min_skew, _ = get_statistics_skew_from_distributed_cliques(
        distributed_cliques,
        global_distribution
    )
    if average_skew_iterations is not None:
        average_skew_iterations.append(average_skew)
    if max_skew_iterations is not None:
        max_skew_iterations.append(max_skew)
    if min_skew_iterations is not None:
        min_skew_iterations.append(min_skew)


def create_cliques_and_get_average_skew(nodes, global_distribution,
                                        max_n_nodes=None):
    distributed_cliques = []
    average_skew_iterations = []
    max_skew_iterations = []
    min_skew_iterations = []

    for node in nodes:
        one_iteration_of_alg4(distributed_cliques, global_distribution, node,
                              average_skew_iterations, max_skew_iterations,
                              min_skew_iterations, max_n_nodes=max_n_nodes)

    return distributed_cliques, [average_skew_iterations, max_skew_iterations,
                                 min_skew_iterations]


def get_global_distribution(nodes, n_classes):
    global_distribution = np.zeros(n_classes)
    for node_distribution in nodes:
        global_distribution += node_distribution
    return np.array(global_distribution)


def get_statistics_skew_from_distributed_cliques(distributed_cliques,
                                                 global_distribution):
    skew_from_each_distributed_clique = []
    for clique in distributed_cliques:
        skew_from_each_distributed_clique.append(skew(clique,
                                                      global_distribution))
    np_skew = np.array(skew_from_each_distributed_clique)
    average_skew = np_skew.mean()
    max_skew = np_skew.max()
    min_skew = np_skew.min()
    std = np_skew.std()
    return average_skew, max_skew, min_skew, std
