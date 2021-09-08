import time

import matplotlib.pyplot as plt
import numpy as np

import functions


def create_nodes(random_generator, n_classes, number_nodes):
    return random_generator.uniform(100, 10001, (number_nodes, n_classes))


def prob(clique):
    clique_distribution = clique.get_clique_distribution()
    return clique_distribution / clique_distribution.sum()


def skew(clique, global_distribution):
    prob_clique = prob(clique)
    prob_global_distribution = global_distribution / global_distribution.sum()
    probabilities_difference = np.abs(prob_clique - prob_global_distribution)
    return probabilities_difference.sum()


def get_global_distribution(nodes, n_classes):
    global_distribution = np.zeros(n_classes)
    for node_distribution in nodes:
        global_distribution += node_distribution
    return global_distribution


def plot_cliques_distribution(conf, alg_type="GA", save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(conf["global_distribution"] / conf["global_distribution"].sum())
    legend = ["D"]
    for index, clique in enumerate(conf["distributed_cliques"]):
        clique_distribution = clique.get_clique_distribution()
        plt.plot(clique_distribution / clique_distribution.sum(), alpha=0.4)
        legend.append("DC" + str(index))
    plt.legend(legend, loc='lower right', ncol=2)

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}{alg_type}_cliques_distribution"
            f"_{conf['n_nodes']}nodes"
            f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_skew_evolution(conf, alg_type="GA", save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(range(conf["n_nodes"]), conf["average_skew"])
    plt.plot(range(conf["n_nodes"]), conf["max_skew"], color="blue", alpha=0.4)
    plt.plot(range(conf["n_nodes"]), conf["min_skew"], color="blue", alpha=0.4)
    plt.yticks(
        np.arange(conf["min_y_axis"], conf["max_y_axis"], conf["step_y_axis"]))
    plt.grid(True)

    if save:
        plt.savefig(f"{conf['pictures_folder_path']}{alg_type}_skew_evolution_"
                    f"{conf['n_nodes']}nodes_{conf['iterations']}iterations_"
                    f"{conf['max_nodes_per_clique']}maxnodes"
                    f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_incidence_of_nodes_in_average_skew(conf, alg_type="GA", save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.plot(conf["n_nodes_list"], conf["average_skew_list"])
    plt.plot(conf["n_nodes_list"], conf["max_skew_list"], color="blue",
             alpha=0.4)
    plt.plot(conf["n_nodes_list"], conf["GA_min_skew_list"], color="blue",
             alpha=0.4)
    plt.grid(True)
    plt.xticks(
        np.arange(min(conf["n_nodes_list"]), max(conf["n_nodes_list"]) + 1, 1))

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}{alg_type}_n_node_incidence_"
            f"{conf['n_nodes']}nodes_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_skew_distribution_over_cliques(conf, alg_type="GA", save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.hist(conf["cliques_skew_list"])
    plt.grid(True)

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}{alg_type}"
            f"_skew_distribution_over_cliques_"
            f"{conf['n_nodes']}nodes_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_number_combinations_opt_alg(conf, save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.plot(conf["n_nodes_list"], conf["number_combinations"])
    plt.grid(True)
    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}OA_number_combinations"
            f"_{conf['n_nodes_list']}_nodes_{conf['iterations']}iterations_"
            f"{conf['max_nodes_per_clique']}maxnodes"
            f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_time_for_constructing_cliques(conf, save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.plot(conf["n_nodes_list"], conf["GA_time"])
    plt.plot(conf["n_nodes_list"], conf["OA_time"])
    plt.legend(["Greedy Algorithm", "Optimal Algorithm"])
    plt.grid(True)

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}"
            f"GA_OA_time_for_constructing_cliques_"
            f"{conf['n_nodes_list']}_nodes_{conf['iterations']}iterations"
            f"_{conf['max_nodes_per_clique']}maxnodes"
            f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_avg_skew_distributed_cliques(conf, save=False):
    plt.figure().clear()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.plot(conf["n_nodes_list"], conf["GA_avg_skew_cliques"])
    plt.plot(conf["n_nodes_list"], conf["OA_avg_skew_cliques"])
    plt.legend(["Greedy Algorithm", "Optimal Algorithm"])
    plt.grid(True)

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}"
            f"GA_OA_avg_skew_of_distributed_cliques_"
            f"{conf['n_nodes_list']}_nodes_{conf['iterations']}iterations_"
            f"{conf['max_nodes_per_clique']}maxnodes"
            f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()


def plot_skew_distribution_over_cliques(conf, alg_type="OA", save=False):
    try:
        f, a = plt.subplots(conf["subplots_n_r"], conf["subplots_n_c"],
                            figsize=(20, 10))
        a = a.ravel()
        for idx, ax in enumerate(a):
            if idx < len(conf["GA_conf_skew_distribution_list"]):
                ax.hist(conf["GA_conf_skew_distribution_list"][idx][
                            "cliques_skew_list"], alpha=0.8)
                ax.hist(conf["OA_conf_skew_distribution_list"][idx][
                            "cliques_skew_list"], alpha=0.8)
                ax.set_title(conf["OA_conf_skew_distribution_list"][idx][
                                 "title"], fontsize=10)
                ax.set_xlabel(
                    conf["OA_conf_skew_distribution_list"][idx]["x_label"])
                ax.set_ylabel(
                    conf["OA_conf_skew_distribution_list"][idx]["y_label"])

            plt.tight_layout()
            plt.legend(["Greedy Algorithm", "Optimal Algorithm"])

        if save:
            plt.savefig(
                f"{conf['OA_conf_skew_distribution_list'][0]['pictures_folder_path']}"
                f"GA_OA_skew_distribution_over_cliques_{conf['n_nodes']}_nodes"
                f"_{conf['iterations']}iterations"
                f"_{conf['max_nodes_per_clique']}maxnodes_"
                f"s{str(int(time.time()))}.pdf")
        else:
            plt.show()
    except:
        plt.figure().clear()
        plt.title(f"{alg_type} - {conf['n_nodes']} nodes - N/"
                  f"{conf['max_nodes_per_clique']}")
        plt.xlabel(conf["x_label"])
        plt.ylabel(conf["y_label"])
        plt.rc('axes', axisbelow=True)
        plt.hist(conf["cliques_skew_list"])
        # plt.legend(["Greedy Algorithm", "Optimal Algorithm"])
        plt.grid(True)

        if save:
            plt.savefig(
                f"{conf['pictures_folder_path']}{alg_type}"
                f"_skew_distribution_over_cliques_{conf['n_nodes']}_nodes"
                f"_{conf['iterations']}iterations_"
                f"{conf['max_nodes_per_clique']}maxnodes"
                f"_{str(int(time.time()))}.pdf")
        else:
            plt.show()


def plot_individual_skew_distribution_over_cliques(conf, save=False):
    plt.figure()
    plt.title(conf["title"])
    plt.xlabel(conf["x_label"])
    plt.ylabel(conf["y_label"])
    plt.rc('axes', axisbelow=True)
    plt.hist(conf["GA_skew_distribution"])
    plt.hist(conf["OA_skew_distribution"])
    plt.legend(["Greedy Algorithm", "Optimal Algorithm"])
    plt.grid(True)

    if save:
        plt.savefig(
            f"{conf['pictures_folder_path']}"
            f"GA_OA_skew_distribution_over_cliques_"
            f"{conf['n_nodes']}nodes_{conf['iterations']}"
            f"_{conf['max_nodes_per_clique']}maxnodes"
            f"_{str(int(time.time()))}.pdf")
    else:
        plt.show()
