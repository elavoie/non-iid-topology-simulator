import numpy as np
import pickle
import time
from pathlib import Path

import greedy_with_pre_comp_of_D_changed as greedy_funcs
import functions


class DecentralizedClique(greedy_funcs.Clique):
    def __init__(self, node_distribution, approximate_global_distribution,
                 clique_id):
        super().__init__(node_distribution)
        self.approximate_global_distribution = approximate_global_distribution
        self.sum_all_nodes_approximate = approximate_global_distribution
        self.clique_id = clique_id
        self.received_messages_on_this_round = []
        self.received_messages_on_previous_round = []
        self.received_info_about_graph = []  # [[clique_ids] for i in range(len(graph))]
        self.nodes_that_prefer_me = []
        self.current_preferee = None
        self.will_live = False
        self.nodes_ids = [self.clique_id]

    @staticmethod
    def get_cycle_from_graph(graph, clique_id):
        cycle = [clique_id]
        curr = graph[clique_id]
        min_index = 0
        while curr != clique_id:
            cycle.append(curr)
            if curr < cycle[min_index]:
                min_index = len(cycle) - 1
            curr = graph[curr]
        cycle = cycle[min_index:] + cycle[:min_index]
        return cycle, (len(cycle) - min_index) % len(cycle)

    def merge_clique(self, clique):
        self.will_live = True
        for i in range(len(clique.all_nodes)):
            self.add_node(clique.all_nodes[i])
        self.sum_all_nodes_approximate += clique.sum_all_nodes_approximate
        self.approximate_global_distribution = self.sum_all_nodes_approximate / len(
            clique.all_nodes)
        self.nodes_ids += clique.nodes_ids

    def reset_clique(self, preferee, clique_id):
        self.current_preferee = preferee
        self.clique_id = clique_id
        self.will_live = False
        self.received_messages_on_this_round = []
        self.received_messages_on_previous_round = []
        self.received_info_about_graph = [-1] * (self.clique_id + 1)
        self.received_info_about_graph[clique_id] = preferee
        self.nodes_that_prefer_me = []

    def first_communication_round(self, all_cliques):
        if self.current_preferee is None:
            return
        all_cliques[
            self.current_preferee].received_messages_on_previous_round.append(
            (self.clique_id, self.current_preferee))
        all_cliques[self.current_preferee].nodes_that_prefer_me.append(
            self.clique_id)

    def talk_to_nodes_that_prefer_me(self, is_free, prev_node, all_cliques):
        sent_yes = not is_free
        for j in range(len(self.nodes_that_prefer_me)):
            if self.nodes_that_prefer_me[j] != prev_node:
                if not sent_yes:
                    message = (self.clique_id, self.nodes_that_prefer_me[j],
                               "YES")
                    sent_yes = True
                else:
                    message = (self.clique_id, self.nodes_that_prefer_me[j],
                               "NO")
                all_cliques[self.nodes_that_prefer_me[
                    j]].received_messages_on_this_round.append(message)
        if not sent_yes:
            self.will_live = True

    def communication_round(self, all_cliques):
        len_received = len(self.received_messages_on_previous_round)
        start_time = time.time()
        if self.current_preferee is None:
            self.talk_to_nodes_that_prefer_me(True, -1, all_cliques)
            return False, time.time() - start_time, len_received
        for i in range(len(self.received_messages_on_previous_round)):
            from_node = self.received_messages_on_previous_round[i][0]
            to_node = self.received_messages_on_previous_round[i][1]
            len_message = len(self.received_messages_on_previous_round[i])
            if len_message == 2:
                if from_node >= len(self.received_info_about_graph):
                    diff = from_node - len(self.received_info_about_graph) + 1
                    self.received_info_about_graph += [-1] * diff
                self.received_info_about_graph[from_node] = to_node
            if from_node == self.current_preferee and len_message == 2:
                cycle, ind = self.get_cycle_from_graph(
                    self.received_info_about_graph,
                    self.clique_id)
                prev_node = cycle[(ind - 1) % len(cycle)]
                if len(cycle) % 2 == 0 or ind != len(cycle) - 1:
                    if ind % 2 == 1:
                        node_to_merge_with = cycle[(ind - 1) % len(cycle)]
                    else:
                        node_to_merge_with = cycle[(ind + 1) % len(cycle)]
                        self.merge_clique(all_cliques[node_to_merge_with])
                else:
                    node_to_merge_with = None
                self.talk_to_nodes_that_prefer_me(node_to_merge_with is None,
                                                  prev_node, all_cliques)
                return False, time.time() - start_time, len_received
            elif from_node == self.current_preferee and len_message == 3:
                decision = self.received_messages_on_previous_round[i][2]
                if decision == "YES":
                    self.merge_clique(all_cliques[from_node])
                self.talk_to_nodes_that_prefer_me(decision == "NO", -1,
                                                  all_cliques)
                return False, time.time() - start_time, len_received
        all_cliques[
            self.current_preferee].received_messages_on_this_round += self.received_messages_on_previous_round
        return True, time.time() - start_time, len_received

    def finish_round(self):
        self.received_messages_on_previous_round = self.received_messages_on_this_round
        self.received_messages_on_this_round = []


def choose_preferred_clique_greedily(i, neighbours, all_cliques,
                                     max_n_nodes=None):
    """
    :param i: index of the clique choosing the preferred one right now
    :param neighbours: random sample where the node all_cliques[i] can look,
        given in the format of indices of all_cliques
    :param all_cliques: a list of DecentralizedClique elements
    :param max_n_nodes: a constraint on the number of nodes in a clique
    :return: index of a chosen clique in all_cliques, None if it is best to not
        form a clique
    """
    neighbours_cliques = []
    for j in range(len(neighbours)):
        neighbours_cliques.append(all_cliques[neighbours[j]])
    if max_n_nodes is not None:
        max_n_nodes = max_n_nodes - len(all_cliques[i].all_nodes)
    index, skew = greedy_funcs.one_iteration_of_alg4(neighbours_cliques,
                                                     all_cliques[
                                                         i].approximate_global_distribution,
                                                     all_cliques[
                                                         i].clique_distribution,
                                                     return_index_not_add=True,
                                                     max_n_nodes=max_n_nodes)
    if skew >= 0:
        return None
    return neighbours[index]


def get_sizes_of_cliques(all_cliques):
    sizes = []
    for i in range(len(all_cliques)):
        sizes.append(len(all_cliques[i].all_nodes))
    return [min(sizes), sum(sizes) / len(sizes), max(sizes),
            np.array(sizes).std()]


def decentralized_greedy_resolving_conflicts(all_cliques, rng,
                                             global_distribution,
                                             number_of_components=None,
                                             size_of_components=None,
                                             len_of_cycles=None,
                                             number_of_needed_iterations=None,
                                             mean_time_for_iteration=None,
                                             number_of_messages=None,
                                             number_of_cliques=None,
                                             size_of_cliques=None,
                                             directed="",
                                             random_sample=10,
                                             iterations=10,
                                             max_n_nodes=None):
    average_skew_iterations = []
    curr_times = []
    for _ in range(iterations):
        number_of_nodes = len(all_cliques)
        if number_of_cliques is not None:
            number_of_cliques.append(number_of_nodes)
        if number_of_nodes == 1:
            if number_of_components is not None:
                number_of_components.append(1)
            if size_of_components is not None:
                size_of_components.append([1, 1, 1])
            if len_of_cycles is not None:
                len_of_cycles.append([[0, 0, 0], [0, 0, 0]])
            if number_of_needed_iterations is not None:
                number_of_needed_iterations.append(0)
            if mean_time_for_iteration is not None:
                mean_time_for_iteration.append([0, 0, 0])
            if number_of_messages is not None:
                number_of_messages.append(0)
            if size_of_cliques is not None:
                size_of_cliques.append([len(all_cliques[0].all_nodes)] * 3 +
                                       [0])
            break
        if size_of_cliques is not None:
            size_of_cliques.append(get_sizes_of_cliques(all_cliques))
        num_of_neighbours = min(random_sample, number_of_nodes - 1)
        graph = functions.build_graph(number_of_nodes, "RANDOM" + directed,
                                      num_of_neighbours, rng)
        for i in range(number_of_nodes):
            all_cliques[i].reset_clique(
                choose_preferred_clique_greedily(i, graph[i], all_cliques,
                                                 max_n_nodes), i)
        if number_of_messages is not None:
            if directed == "":
                cnt_messages = len(all_cliques) * num_of_neighbours
            else:
                cnt_messages = 2 * len(all_cliques) * num_of_neighbours
        if number_of_components is not None or size_of_components is not None \
                or len_of_cycles is not None:
            conflicts_graph = [[] for _ in range(number_of_nodes)]
            for i in range(number_of_nodes):
                if all_cliques[i].current_preferee is not None:
                    conflicts_graph[i].append(all_cliques[i].current_preferee)
                    conflicts_graph[all_cliques[i].current_preferee].append(i)
            components, visited, num_cycles = functions.get_number_of_components(
                conflicts_graph)
            if number_of_components is not None:
                number_of_components.append(components)
            if size_of_components is not None:
                sizes = np.zeros(components)
                for i in range(len(visited)):
                    sizes[visited[i]] += 1
                size_of_components.append([sizes.min(), sizes.mean(),
                                           sizes.max()])
            if len_of_cycles is not None:
                len_cycles = np.array(num_cycles)
                if size_of_components is None:
                    sizes = np.zeros(components)
                    for i in range(len(visited)):
                        sizes[visited[i]] += 1
                percentage_cycles = len_cycles / sizes
                len_of_cycles.append([[len_cycles.min(), len_cycles.mean(),
                                       len_cycles.max()], [percentage_cycles.min(),
                                                    percentage_cycles.mean(),
                                                    percentage_cycles.max()]])
        for i in range(number_of_nodes):
            all_cliques[i].first_communication_round(all_cliques)
        working_nodes = all_cliques
        new_cliques = []
        next_working_nodes = []
        if number_of_needed_iterations is not None:
            cnt_iters = 0
        while working_nodes:
            for i in range(len(working_nodes)):
                is_working, time_node, messages = working_nodes[
                    i].communication_round(all_cliques)
                if is_working:
                    next_working_nodes.append(working_nodes[i])
                if mean_time_for_iteration is not None:
                    curr_times.append(time_node)
                if number_of_messages is not None:
                    cnt_messages += messages
            for i in range(len(working_nodes)):
                working_nodes[i].finish_round()
            working_nodes = next_working_nodes
            next_working_nodes = []
            if number_of_needed_iterations is not None:
                cnt_iters += 1
        if number_of_needed_iterations is not None:
            number_of_needed_iterations.append(cnt_iters)
        if number_of_messages is not None:
            number_of_messages.append(cnt_messages)
        if mean_time_for_iteration is not None:
            mean_time = sum(curr_times) / len(curr_times)
            mean_time_for_iteration.append([min(curr_times), mean_time, max(curr_times)])
        for i in range(len(all_cliques)):
            if all_cliques[i].will_live:
                new_cliques.append(all_cliques[i])
        all_cliques = new_cliques
        average_skew, max_skew, min_skew, std = \
            greedy_funcs.get_statistics_skew_from_distributed_cliques(
                all_cliques, global_distribution)
        average_skew_iterations.append([min_skew, average_skew, max_skew, std])
        if number_of_cliques is not None and len(number_of_cliques) == \
                iterations:
            number_of_cliques.append(len(all_cliques))
        if size_of_cliques is not None and len(size_of_cliques) == iterations:
            size_of_cliques.append(get_sizes_of_cliques(all_cliques))
    return all_cliques, average_skew_iterations


def create_nodes_distributions(N_CLASSES, N_NODES, graph_config,
                               n_shards_per_node, rng, samples_per_class,
                               shard_size):
    result = functions.build_graph_run_pushsum(graph_config, N_CLASSES, rng,
                                               100, shard_size,
                                               samples_per_class,
                                               n_shards_per_node)
    all_nodes, average_skew, _, max_skew, min_skew = result
    N = np.zeros((N_NODES, N_CLASSES))
    for j in range(len(all_nodes)):
        N[j] += all_nodes[j].samples
    return N, all_nodes


def create_cliques_and_compute_global_distribution(N, N_CLASSES, all_nodes,
                                                   decentralized_clique=DecentralizedClique):
    all_cliques = []
    for j in range(len(all_nodes)):
        all_cliques.append(decentralized_clique(all_nodes[j].samples,
                                                all_nodes[
                                                    j].average_global_samples,
                                                j))
    D = greedy_funcs.get_global_distribution(N, N_CLASSES)
    return D, N, all_cliques


def prepare_data(N_CLASSES, N_NODES, graph_config, rng,
                 decentralized_clique=DecentralizedClique,
                 shard_size=None, samples_per_class=None,
                 n_shards_per_node=None):
    N, all_nodes = create_nodes_distributions(N_CLASSES, N_NODES, graph_config,
                                              n_shards_per_node, rng,
                                              samples_per_class, shard_size)
    return create_cliques_and_compute_global_distribution(N, N_CLASSES,
                                                          all_nodes,
                                                          decentralized_clique)


def run_experiments(N_NODES, N_CLASSES, TESTS, rng, iterations, random_sample,
                    statistics_names_without_skew,
                    decentralized_greedy=
                    decentralized_greedy_resolving_conflicts,
                    decentralized_clique=DecentralizedClique, directed="",
                    shard_size=None, samples_per_class=None,
                    n_shards_per_node=None):
    STATISTICS = len(statistics_names_without_skew) + 1
    statistics = [[[] for _ in range(TESTS)] for __ in range(STATISTICS)]
    for i in range(TESTS):
        # print(i)
        graph_config = {"n": N_NODES, "topology": "RANDOM" + directed,
                        "m": random_sample, "rng": rng}
        D, N, all_cliques = prepare_data(N_CLASSES, N_NODES, graph_config, rng,
                                         decentralized_clique=
                                         decentralized_clique,
                                         shard_size=shard_size,
                                         samples_per_class=samples_per_class,
                                         n_shards_per_node=n_shards_per_node)
        decentralized_greedy_parameters = {"all_cliques": all_cliques,
                                           "rng": rng,
                                           "global_distribution": D,
                                           "directed": directed,
                                           "iterations": iterations,
                                           "random_sample": random_sample}
        for j in range(len(statistics_names_without_skew) - 1):
            decentralized_greedy_parameters[statistics_names_without_skew[j]] \
                = statistics[j][i]
        decentralized_greedy_parameters[statistics_names_without_skew[-1]] =\
            statistics[-1][i]
        _, statistics[-2][i] = decentralized_greedy(
            **decentralized_greedy_parameters)
    return statistics


def pad_and_dump_number_and_size(tests, iterations, max_length_statistics,
                                 classes_range, random_sample_range,
                                 nodes_range, statistics, name_prefix,
                                 directed, statistics_path, sharded=False):
    if tests > 1: # we have to pad data to save it in a numpy.array
        for class_index in range(len(classes_range)):
            for random_sample_index in range(len(random_sample_range)):
                for nodes_index in range(len(nodes_range)):
                    for k in range(tests):
                        for a in range(len(statistics)):
                            curr_len = len(statistics[a][class_index][
                                               random_sample_index][
                                               nodes_index][k])
                            addition = [statistics[a][class_index][
                                            random_sample_index][nodes_index][
                                            k][-1] for _ in range(
                                max_length_statistics[a] - curr_len)]
                            statistics[a][class_index][random_sample_index][
                                nodes_index][k] += addition
    for a in range(len(statistics)):
        statistics[a] = np.array(statistics[a])
        parent_folder = "../data/" + statistics_path[a]
        Path(parent_folder).mkdir(parents=True, exist_ok=True)
        filename = parent_folder + f"/{name_prefix}_iterations_" \
                                   f"{str(iterations)}{directed}_sharded_" \
                                   f"{str(sharded)}"
        with open(filename, "wb") as file:
            pickle.dump(statistics[a], file)


def process_result(class_index, random_sample_index, nodes_index,
                   max_length_statistics, result, statistics):
    assert len(result) == len(statistics)
    for k in range(len(result)):
        statistics[k][class_index][random_sample_index][nodes_index] = result[
            k]
        max_length_statistics[k] = max(max_length_statistics[k],
                                       max(map(len, result[k])))


def dump_data_for_classes_random_samples_nodes(rng,
                                               classes_range,
                                               random_sample_range,
                                               nodes_range,
                                               directed,
                                               statistics_paths,
                                               statistics_names_without_skew,
                                               tests=3,
                                               iterations=10,
                                               filename="from_classes",
                                               decentralized_greedy=
                                               decentralized_greedy_resolving_conflicts,
                                               decentralized_clique=
                                               DecentralizedClique,
                                               print_progr=False,
                                               shard_size=None,
                                               samples_per_class=None,
                                               n_shards_per_node=None):
    STATISTICS = len(statistics_paths)
    statistics = [[[[0] * len(nodes_range) for _ in range(len(
        random_sample_range))] for __ in range(len(classes_range))]
                  for ___ in range(STATISTICS)]
    max_length_statistics = [-1] * len(statistics)
    for class_index in range(len(classes_range)):
        for random_sample_index in range(len(random_sample_range)):
            for nodes_index in range(len(nodes_range)):
                if print_progr:
                    print(class_index, random_sample_index, nodes_index)
                result = run_experiments(nodes_range[nodes_index],
                                         classes_range[class_index],
                                         tests,
                                         rng,
                                         iterations=iterations,
                                         random_sample=random_sample_range[
                                             random_sample_index],
                                         directed=directed,
                                         statistics_names_without_skew=
                                         statistics_names_without_skew,
                                         decentralized_greedy=
                                         decentralized_greedy,
                                         decentralized_clique=
                                         decentralized_clique,
                                         shard_size=shard_size,
                                         samples_per_class=samples_per_class,
                                         n_shards_per_node=n_shards_per_node)
                process_result(class_index, random_sample_index, nodes_index,
                               max_length_statistics, result, statistics)
    pad_and_dump_number_and_size(tests, iterations, max_length_statistics,
                                 classes_range, random_sample_range,
                                 nodes_range, statistics, filename, directed,
                                 statistics_paths, sharded=(shard_size is not
                                                            None))


if __name__ == "__main__":
    statistics_names = ["number_of_components", "size_of_components",
                        "len_of_cycles",
                        "number_of_needed_iterations",
                        "mean_time_for_iteration",
                        "number_of_messages",
                        "number_of_cliques",
                        "size_of_cliques"]
    statistics_paths = []
    for i in range(len(statistics_names)):
        statistics_paths.append("test_" + statistics_names[i])
    statistics_paths = statistics_paths[:-1] + [
        "test_average_skew_iterations"] + statistics_paths[-1:]
    # configurations = [{"classes_range": [1, 2, 5, 10, 100, 1000],
    #                    "random_sample_range": [10],
    #                    "nodes_range": [50, 100, 600, 1000],
    #                    "directed": "",
    #                    "statistics_path": statistics_names,
    #                    "iterations": 10},
                     #  {"classes_range": [10],
                     #   "random_sample_range": [10],
                     #   "nodes_range": [20, 50, 75, 100, 250, 500,
                     #                   750, 1000, 1500],
                     #   "directed": "",
                     #   "statistics_path": statistics_names,
                     #   "iterations": 1},
                     #  {"classes_range": [2, 3, 4, 5, 6, 10, 50,
                     #                     100, 500, 1000],
                     #   "random_sample_range": [10],
                     #   "nodes_range": [100, 1000],
                     #   "directed": "",
                     #   "statistics_path": statistics_names,
                     #   "iterations": 10,
                     #   "filename": "max_from_classes"},
                     #  {"classes_range": [10],
                     #   "random_sample_range": [1, 2, 5, 10,
                     #                          19],
                     #   "nodes_range": [20, 50, 80, 100, 250,
                     #                  500, 750, 1000,
                     #                  1500],
                     #   "directed": "",
                     #   "statistics_path": statistics_names,
                     #   "iterations": 1},
                     # {"classes_range": [10],
                     #   "random_sample_range": [2, 6, 10,
                     #                          18],
                     #   "nodes_range": [20, 50, 80, 100, 250,
                     #                  500, 750, 1000,
                     #                  1500],
                     #   "directed": "",
                     #   "statistics_names": statistics_names,
                     #   "iterations": 10}
                    # ]
    # configurations = [
                       # {"classes_range": [1, 2],
                       # "random_sample_range": [10],
                       # "nodes_range": [50, 100],
                       # "directed": "",
                       # "statistics_path": statistics_names,
                       # "iterations": 10},
                       # {"classes_range": [10],
                       #  "random_sample_range": [10],
                       #  "nodes_range": [20, 50],
                       #  "directed": "",
                       #  "statistics_path": statistics_names,
                       #  "iterations": 1},
                       # {"classes_range": [2, 3],
                       #  "random_sample_range": [10],
                       #  "nodes_range": [100, 1000],
                       #  "directed": "",
                       #  "statistics_path": statistics_names,
                       #  "iterations": 10,
                       #  "filename": "max_from_classes"},
                      #  {"classes_range": [10],
                      #   "random_sample_range": [1, 2],
                      #   "nodes_range": [20, 50],
                      #   "directed": "",
                      #   "statistics_paths": statistics_paths,
                      #   "statistics_names_without_skew": statistics_names,
                      #   "iterations": 1,
                      #   "filename": "from_random_sample"},
                      # {"classes_range": [10],
                      #   "random_sample_range": [2, 6],
                      #   "nodes_range": [20, 50],
                      #   "directed": "",
                      #   "statistics_paths": statistics_paths,
                      #   "statistics_names_without_skew": statistics_names,
                      #   "iterations": 10,
                      #   "filename": "from_random_sample"}
                      # ]
    configurations = [{"classes_range": [10],
                       "nodes_range": [100],
                       "random_sample_range": [2],
                       "statistics_paths": statistics_paths,
                       "statistics_names_without_skew": statistics_names,
                       "directed": "",
                       "iterations": 10,
                       "shard_size": 300,
                       "samples_per_class": 6000,
                       "print_progr": True
                       }]
    for conf in range(len(configurations)):
        configurations[conf]["rng"] = np.random.default_rng(24)
        dump_data_for_classes_random_samples_nodes(**configurations[conf])
        configurations[conf]["rng"] = np.random.default_rng(24)
        configurations[conf]["directed"] = "_DIRECTED"
        dump_data_for_classes_random_samples_nodes(**configurations[conf])
