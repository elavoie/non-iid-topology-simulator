import copy
import time

import numpy as np
import matplotlib.pyplot as plt


class Clique:
    def __init__(self, node_distribution):
        self.all_nodes = [copy.deepcopy(node_distribution)]
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
    return global_distribution


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


def plot_cliques_distribution(conf, save=False):
    plt.figure(1)
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(conf["global_distribution"] / conf["global_distribution"].sum())
    legend = ["D"]
    for i, clique in enumerate(conf["distributed_cliques"]):
        clique_distribution = clique.get_clique_distribution()
        plt.plot(clique_distribution / clique_distribution.sum(), alpha=0.4)
        legend.append("DC" + str(i))
    plt.legend(legend, loc='lower right', ncol=2)

    if save:
        plt.savefig("../test_pictures/cliques_distribution_" + str(int(time.time())) + ".pdf")
    else:
        plt.show()


def plot_skew_evolution(conf, save=False):
    plt.figure(2)
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(range(conf["n_nodes"]), conf["average_skew"])
    plt.plot(range(conf["n_nodes"]), conf["max_skew"], color="blue", alpha=0.4)
    plt.plot(range(conf["n_nodes"]), conf["min_skew"], color="blue", alpha=0.4)
    plt.yticks(np.arange(conf["min_y_axis"], conf["max_y_axis"], conf["step_y_axis"]))
    plt.grid(True)

    if save:
        plt.savefig(f"{conf['pictures_folder_path']}skew_evolution_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_incidence_of_nodes_in_average_skew(conf, save=False):
    plt.figure(3)
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(conf["n_nodes_list"], conf["average_skew_list"])
    plt.plot(conf["n_nodes_list"], conf["max_skew_list"], color="blue", alpha=0.4)
    plt.plot(conf["n_nodes_list"], conf["min_skew_list"], color="blue", alpha=0.4)
    plt.grid(True)
    plt.xticks(np.arange(min(n_nodes_list), max(n_nodes_list) + 1, 1))

    if save:
        plt.savefig(f"{conf['pictures_folder_path']}n_node_incidence_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_skew_distribution_over_cliques(conf, save=False):
    plt.figure(4)
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.hist(conf["cliques_skew_list"])
    plt.grid(True)

    if save:
        plt.savefig(f"{conf['pictures_folder_path']}skew_distribution_over_cliques"
                    f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    N_CLASSES = 10
    N_NODES = 600
    RANDOM_GENERATOR = np.random.default_rng(70)
    PICTURES_FOLDER_PATH = "../test_pictures/"
    PLOT = False
    SAVE_PLOTS = False

    N = create_nodes(RANDOM_GENERATOR, N_CLASSES, N_NODES)
    D = get_global_distribution(N, N_CLASSES)
    DC, skew_iterations_results = create_cliques_and_get_average_skew(N, D)
    cliques_distribution_conf = {
        "title": "Cliques distribution",
        "x_label": "Classes",
        "y_label": "Distribution",
        "distributed_cliques": DC,
        "global_distribution": D
    }

    average_skew_list, max_skew_list, min_skew_list = skew_iterations_results
    skew_evolution_conf = {
        "title": "Skew evolution through nodes addition",
        "x_label": "Number of nodes",
        "y_label": "Average skew",
        "average_skew": average_skew_list,
        "max_skew": max_skew_list,
        "min_skew": min_skew_list,
        "min_y_axis": 0.00,
        "max_y_axis": 0.95,
        "step_y_axis": 0.05,
        "n_nodes": N_NODES,
        "pictures_folder_path": PICTURES_FOLDER_PATH
    }

    print(N_NODES, average_skew_list[-1])
    #
    #
    # n_nodes_and_skew_dict = dict()
    # cliques_skew_list = []
    # for i, C in enumerate(DC):
    #     n_nodes = len(C.all_nodes)
    #     clique_skew = skew(C, D)
    #     cliques_skew_list.append(clique_skew)
    #
    #     if str(n_nodes) not in n_nodes_and_skew_dict:
    #         n_nodes_and_skew_dict[str(n_nodes)] = [clique_skew]
    #     else:
    #         n_nodes_and_skew_dict[str(n_nodes)].append(clique_skew)
    #
    # n_nodes_list = np.array([int(number) for number in list(n_nodes_and_skew_dict.keys())])
    # skew_average = []
    # skew_max = []
    # skew_min = []
    #
    # for skew_array in n_nodes_and_skew_dict.values():
    #     skew_average.append(np.array(skew_array).mean())
    #     skew_max.append(max(skew_array))
    #     skew_min.append(min(skew_array))
    #
    # sorted_n_nodes_index = np.argsort(n_nodes_list)
    #
    # nodes_incidence_conf = {
    #     "title": "Incidence of number of nodes a clique over average skew",
    #     "x_label": "Number of nodes added",
    #     "y_label": "Average skew",
    #     "n_nodes_list": n_nodes_list[sorted_n_nodes_index],
    #     "average_skew_list": np.array(skew_average)[sorted_n_nodes_index],
    #     "max_skew_list": np.array(skew_max)[sorted_n_nodes_index],
    #     "min_skew_list": np.array(skew_min)[sorted_n_nodes_index],
    #     "pictures_folder_path": PICTURES_FOLDER_PATH
    # }
    #
    # skew_distribution_conf = {
    #     "title": "",
    #     "x_label": "Skew",
    #     "y_label": "Number of cliques",
    #     "cliques_skew_list": cliques_skew_list,
    #     "pictures_folder_path": PICTURES_FOLDER_PATH
    # }
    #
    # if PLOT:
    #     plot_cliques_distribution(cliques_distribution_conf, SAVE_PLOTS)
    #     plot_skew_evolution(skew_evolution_conf, SAVE_PLOTS)
    #     plot_incidence_of_nodes_in_average_skew(nodes_incidence_conf, SAVE_PLOTS)
    #     plot_skew_distribution_over_cliques(skew_distribution_conf, SAVE_PLOTS)
