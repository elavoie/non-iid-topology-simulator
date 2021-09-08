import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import utils
import functions
import constants


class GAClique:
    def __init__(self, node_distribution):
        self.all_nodes = [copy.deepcopy(node_distribution)]
        self.clique_distribution = self.all_nodes[-1]

    def update_clique_distribution(self):
        self.clique_distribution += self.all_nodes[-1]

    def add_node(self, node_distribution):
        self.all_nodes.append(node_distribution)
        self.update_clique_distribution()
        return self

    def get_clique_distribution(self):
        return self.clique_distribution

    def get_n_nodes_in_clique(self):
        return len(self.all_nodes)


def create_cliques_and_get_skew_statistics(nodes, global_distribution,
                                           max_n_nodes=None):
    distributed_cliques = []

    for node in nodes:
        if distributed_cliques:
            lowest_skew = 0
            index_of_lowest_skew_between_cliques = -1
            for _index in range(len(distributed_cliques)):
                clique = distributed_cliques[_index]
                if max_n_nodes is None or clique.get_n_nodes_in_clique() < \
                        max_n_nodes:
                    temporary_clique = copy.deepcopy(clique).add_node(node)
                    temporary_clique_skew = utils.skew(temporary_clique,
                                                       global_distribution)
                    clique_skew = utils.skew(clique, global_distribution)
                    curr_diff = temporary_clique_skew - clique_skew
                    if index_of_lowest_skew_between_cliques < 0 or curr_diff \
                            < lowest_skew:
                        index_of_lowest_skew_between_cliques = _index
                        lowest_skew = curr_diff
                    del temporary_clique

            if lowest_skew >= 0:
                distributed_cliques.append(GAClique(node))
            else:
                distributed_cliques[
                    index_of_lowest_skew_between_cliques].add_node(node)

        else:
            distributed_cliques.append(GAClique(node))

    average_skew, max_skew, min_skew = \
        get_statistics_skew_from_distributed_cliques(
            distributed_cliques,
            global_distribution
        )

    return distributed_cliques, [average_skew, max_skew, min_skew]


def greedy_algorithm(n, d, max_nodes_per_cliques):
    """
    :param n: list of nodes in the system
    :param d: global distribution of the system
    :param max_nodes_per_cliques: lists of constraints
    :return: Tuple with distributed cliques list and a list of lists about skew
    statistics: average skew list, minimum skew list and maximum skew list
    respectively
    """
    _greedy_distributed_cliques_skew_statistics = [[], [], []]
    dc_all_constraints = []
    for max_nodes_per_clique in max_nodes_per_cliques:
        dc, skew_statistics = create_cliques_and_get_skew_statistics(
            n, d, max_nodes_per_clique)
        for _index in range(len(skew_statistics)):
            _greedy_distributed_cliques_skew_statistics[_index].append(
                skew_statistics[_index])
        dc_all_constraints.append(dc)
    return dc_all_constraints, _greedy_distributed_cliques_skew_statistics


def get_statistics_skew_from_distributed_cliques(distributed_cliques,
                                                 global_distribution):
    skew_from_each_distributed_clique = []
    for clique in distributed_cliques:
        skew_from_each_distributed_clique.append(
            utils.skew(clique, global_distribution))
    average_skew = np.array(skew_from_each_distributed_clique).mean()
    max_skew = max(np.array(skew_from_each_distributed_clique))
    min_skew = min(np.array(skew_from_each_distributed_clique))

    return average_skew, max_skew, min_skew


if __name__ == "__main__":
    RANDOM_GENERATOR = np.random.default_rng(69)
    PLOT_TOP_Y_LIMIT = 1.90
    PLOT_NAME = "skew_size_of_clique_greedy"
    PICTURES_FOLDER_PATH = \
        "../pictures/centralized_greedy/average_skew_iterations/"

    statistics = np.array(
        [np.zeros((constants.ExperimentSetting.TESTS,
                  len(constants.Constraints.MAX_NODES_PER_CLIQUE)))
         for _ in range(3)])
    for i in range(constants.ExperimentSetting.TESTS):
        N = None
        if constants.ExperimentSetting.DATA_PARTITION_MODE == "simple":
            N = utils.create_nodes(RANDOM_GENERATOR,
                                   constants.ExperimentSetting.N_CLASSES,
                                   constants.ExperimentSetting.N_NODES)
        elif constants.ExperimentSetting.DATA_PARTITION_MODE == "shards":
            N = functions.data_partition(
                constants.ExperimentSetting.N_NODES,
                constants.ExperimentSetting.N_CLASSES, RANDOM_GENERATOR,
                constants.ExperimentSetting.SHARD_SIZE,
                constants.ExperimentSetting.SAMPLES_PER_CLASS)
        assert N is not None, "N has not been defined, check your partition " \
                              "mode, it should be simple or shards"
        D = utils.get_global_distribution(N,
                                          constants.ExperimentSetting.N_CLASSES)
        DC_all_constraints, greedy_distributed_cliques_skew_statistics = \
            greedy_algorithm(
                N, D, constants.Constraints.MAX_NODES_PER_CLIQUE)
        for index in range(len(statistics)):
            statistics[index][i] = greedy_distributed_cliques_skew_statistics[
                index]

    avg_skew_per_constraint = np.mean(statistics[0], axis=0)
    min_skew_per_constraint = np.max(statistics[1], axis=0)
    max_skew_per_constraint = np.min(statistics[2], axis=0)

    # Plotting statistics
    plt.title("Centralized greedy algorithm")
    plt.xlabel("Max number of nodes in a clique")
    plt.ylabel("Average skew")
    plt.plot(constants.Constraints.MAX_NODES_PER_CLIQUE,
             avg_skew_per_constraint,
             color=constants.Figures.COLOR_PRIMARY)
    plt.plot(constants.Constraints.MAX_NODES_PER_CLIQUE,
             min_skew_per_constraint,
             color=constants.Figures.COLOR_PRIMARY, alpha=0.4)
    plt.plot(constants.Constraints.MAX_NODES_PER_CLIQUE,
             max_skew_per_constraint,
             color=constants.Figures.COLOR_PRIMARY, alpha=0.4)
    if PLOT_TOP_Y_LIMIT:
        axes = plt.gca()
        axes.set_ylim(top=PLOT_TOP_Y_LIMIT)
    plt.grid(True)
    if constants.Figures.SAVE_PLOT:
        # Create a folder if it does not exist
        Path(PICTURES_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"{PICTURES_FOLDER_PATH}{PLOT_NAME}_"
            f"{constants.ExperimentSetting.DATA_PARTITION_MODE}_partition_"
            f"{constants.ExperimentSetting.N_NODES}_nodes.pdf",
            bbox_inches='tight')
    plt.show()
