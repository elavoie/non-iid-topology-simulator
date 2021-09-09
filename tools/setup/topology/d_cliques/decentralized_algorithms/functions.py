import copy
import time

import matplotlib.pyplot as plt
import numpy as np


def plot_times(times, nodes_range):
    means = times.mean(axis=1)
    mins = times.min(axis=1)
    maxs = times.max(axis=1)
    plt.plot(nodes_range, means)
    plt.grid(True)
    plt.fill_between(nodes_range, mins, maxs, color='#888888', alpha=0.4)
    plt.xlabel("Number of nodes")
    plt.ylabel("Time (sec)")
    plt.title("Time of computing the distribution")
    plt.savefig("../pictures/centr_times_" + str(len(nodes_range)) +
                "_points_6_tests.pdf", bbox_inches='tight')
    plt.show()


def build_graph(n, topology, m=None, rng=None, connected=False):
    if topology == "RING":
        graph = [[] for _ in range(n)]
        for i in range(n):
            graph[i].append((i + 1) % n)
            graph[i].append((i - 1) % n)
        return graph
    elif topology == "GRID":
        if m is None:
            raise ValueError("Second dimension for grid is not given")
        graph = [[] for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                curr_node = i * m + j
                if i + 1 < n:
                    graph[curr_node].append((i + 1) * m + j)
                if i - 1 >= 0:
                    graph[curr_node].append((i - 1) * m + j)
                if j + 1 < m:
                    graph[curr_node].append(i * m + j + 1)
                if j - 1 >= 0:
                    graph[curr_node].append(i * m + j - 1)
        return graph
    elif topology == "FC":
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                graph[i].append(j)
                graph[j].append(i)
        return graph
    elif topology == "RANDOM":
        # ListDict copied from
        # https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
        class ListDict(object):
            def __init__(self, items):
                self.item_to_position = {}
                self.items = items
                for i in range(len(items)):
                    self.item_to_position[items[i]] = i

            def remove_item(self, item):
                position = self.item_to_position.pop(item)
                last_item = self.items.pop()
                if position != len(self.items):
                    self.items[position] = last_item
                    self.item_to_position[last_item] = position

            def choose_random_items(self, rng, n):
                return list(rng.choice(self.items, n, replace=False))

            def size(self):
                return len(self.items)

        if m is None:
            raise ValueError("Number of neighbours for random graph is not "
                             "given")
        if n <= m:
            raise ValueError(
                "The number of neighbours is bigger than the number of nodes")
        if n % 2 == 1 and m % 2 == 1:
            raise ValueError(f"There cannot be a graph of {str(n)} vertices "
                             f"with {str(m)} neighbours each")
        if rng is None:
            raise ValueError("Random generator is not given")
        graph_finished = False
        cnt = 0
        while not graph_finished:
            cnt += 1
            graph = [[] for _ in range(n)]
            available_vertices = ListDict(list(range(n)))
            for i in range(n):
                if len(graph[i]) < m:
                    available_vertices.remove_item(i)
                    if available_vertices.size() < m - len(graph[i]):
                        break
                    random_neighbours = available_vertices.choose_random_items(
                        rng, m - len(graph[i]))
                    graph[i] += random_neighbours
                    for j in range(len(random_neighbours)):
                        graph[random_neighbours[j]].append(i)
                        if len(graph[random_neighbours[j]]) == m:
                            available_vertices.remove_item(
                                random_neighbours[j])
            else:
                if connected and get_number_of_components(graph)[0] > 1:
                    graph_finished = False
                else:
                    graph_finished = True
        return graph
    elif topology == "RANDOM_DIRECTED":
        if m is None:
            raise ValueError("Number of neighbours for random graph is not "
                             "given")
        if n <= m:
            raise ValueError(
                "The number of neighbours is bigger than the number of nodes")
        if rng is None:
            raise ValueError("Random generator is not given")
        graph = [0] * n
        for i in range(n):
            graph[i] = rng.choice(list(range(i)) + list(range(i + 1, n)), m,
                                  replace=False)
        return graph
    raise ValueError("Topology " + topology + " is not implemented.")


class Node:
    def __init__(self, rng, neighbours, classes, starting_node=0,
                 send_all_neighbours=False, samples=None):
        self.rng = rng
        self.neighbours = neighbours
        if samples is None:
            self.samples = rng.uniform(100, 10001, classes)
        else:
            self.samples = samples
        self.prev_info = [(self.samples, starting_node)]
        self.new_info = []
        self.average_global_samples = np.zeros(classes)
        self.cnt_rounds = 0
        self.time_by_iteration = []
        self.send_all_neighbours = send_all_neighbours

    def communication_round(self, all_nodes):
        start_time = time.perf_counter()
        # 1: compute the sum of previous messages
        s, w = 0, 0
        for i in range(len(self.prev_info)):
            s += self.prev_info[i][0]
            w += self.prev_info[i][1]
        # 2: compute the current estimate
        self.average_global_samples = s / w
        # 3: compute the message
        if self.send_all_neighbours:
            n = len(self.neighbours) + 1
            message = (s / n, w / n)
        else:
            message = (s / 2, w / 2)
        self.time_by_iteration.append(time.perf_counter() - start_time)
        if self.send_all_neighbours:
            for i in range(len(self.neighbours)):
                all_nodes[self.neighbours[i]].new_info.append(message)
            self.new_info.append(message)
        else:
            random_neighbour = self.rng.choice(self.neighbours)
            all_nodes[random_neighbour].new_info.append(message)
            self.new_info.append(message)
        self.cnt_rounds += 1
        if self.send_all_neighbours:
            return len(self.neighbours)
        else:
            return 1

    def finish_round(self):
        self.prev_info = self.new_info
        self.new_info = []


def get_title(configurations, conf):
    title = configurations[conf]["topology"]
    if title == "RANDOM":
        title += ", " + str(configurations[conf]["m"])
    elif title == "GRID":
        title += ", " + str(configurations[conf]["n"]) + "x" + str(
            configurations[conf]["m"])
    return title


def mean_skew_N(N, CLASSES, TESTS, rng):
    uniform_distribution = np.ones(CLASSES) / CLASSES
    skew = np.zeros(TESTS)
    for i in range(TESTS):
        distribution = rng.uniform(100, 10001, (N, CLASSES)).sum(axis=0)
        distribution /= distribution.sum()
        skew[i] = np.abs(distribution - uniform_distribution).sum()
    return skew.mean()


def build_graph_run_pushsum(configuration, CLASSES, rng, iterations,
                            shard_size=None, samples_per_class=None,
                            n_shards_per_node=None):
    graph = build_graph(**configuration)
    all_nodes = []
    distribution = np.zeros(CLASSES)
    if shard_size is not None:
        samples = data_partition(len(graph), CLASSES, rng, shard_size,
                                 samples_per_class, n_shards_per_node)
    for i in range(len(graph)):
        if shard_size:
            node = Node(rng, graph[i], CLASSES, starting_node=True,
                        samples=samples[i])
        else:
            node = Node(rng, graph[i], CLASSES, starting_node=True)
        all_nodes.append(node)
        distribution += all_nodes[-1].samples
    distribution /= distribution.sum()
    average_skew = []
    max_skew, min_skew = [], []
    for _ in range(iterations):
        curr_skew = 0
        curr_max_skew = -1
        curr_min_skew = CLASSES + 1
        for i in range(len(graph)):
            all_nodes[i].communication_round(all_nodes)
            skew = np.abs(all_nodes[i].average_global_samples / all_nodes[
                i].average_global_samples.sum() - distribution).sum()
            curr_skew += skew
            curr_max_skew = max(curr_max_skew, skew)
            curr_min_skew = min(curr_min_skew, skew)
        curr_skew /= len(graph)
        average_skew.append(curr_skew)
        max_skew.append(curr_max_skew)
        min_skew.append(curr_min_skew)
        for i in range(len(graph)):
            all_nodes[i].finish_round()
    return all_nodes, average_skew, graph, max_skew, min_skew


def dfs(s, prev, graph, visited, colour, path_length):
    visited[s] = colour
    cycle_len = 0
    edge_prev = True
    for i in range(len(graph[s])):
        if graph[s][i] == prev and edge_prev:
            edge_prev = False
            continue
        if visited[graph[s][i]] >= 0:
            cycle_len = max(cycle_len, path_length[s] - path_length[
                graph[s][i]] + 1)
        else:
            path_length[graph[s][i]] = path_length[s] + 1
            cycle_len = max(cycle_len, dfs(graph[s][i], s, graph, visited,
                                           colour, path_length))
    return cycle_len


def get_number_of_components(conflicts_graph):
    visited = [-1] * len(conflicts_graph)
    path_length = [0] * len(conflicts_graph)
    cycles = []
    components = 0
    for i in range(len(visited)):
        if visited[i] < 0:
            cycles.append(dfs(i, -1, conflicts_graph, visited, components,
                              path_length))
            components += 1
    return components, visited, cycles


def data_partition(n_nodes, n_classes, rng, shard_size, _samples_per_class,
                   n_shards_per_node=None):
    """
    :param _samples_per_class: can be int or a list
    :param n_shards_per_node:
    :param shard_size:
    :param n_nodes:
    :param n_classes:
    :param rng: random generator
    :return: list with nodes distribution
    """
    total_samples = None

    if isinstance(_samples_per_class, int):
        total_samples = n_classes * _samples_per_class
        samples_per_class = [copy.deepcopy(_samples_per_class)] * n_classes
    else:
        total_samples = sum(_samples_per_class)
        samples_per_class = copy.deepcopy(_samples_per_class)

    assert total_samples % shard_size == 0, f"total_samples is not divisible " \
                                            "by shard_size"
    n_shards = total_samples // shard_size

    assert n_shards % n_nodes == 0, "n_shards is not divisible by n_nodes"
    if n_shards_per_node:
        assert n_nodes * n_shards_per_node == \
               n_shards, f"n_nodes * n_shards_per_node should be equal to " \
                         f"n_shards={n_shards} to use all data"
    else:
        n_shards_per_node = n_shards // n_nodes

    nodes_distribution = []
    shards_info = np.zeros((n_shards, n_classes), dtype=int)

    # populating shards_info
    class_index = 0
    for shard_index in range(n_shards):
        current_shard_size = 0
        while current_shard_size < shard_size:
            while samples_per_class[class_index] == 0:
                class_index += 1
            needed_samples = shard_size - current_shard_size
            available_class_samples = min(needed_samples, samples_per_class[
                class_index])
            shards_info[shard_index][class_index] += available_class_samples
            samples_per_class[class_index] -= available_class_samples
            current_shard_size += available_class_samples

    # populating nodes_distribution
    rng.shuffle(shards_info)
    shard_counter = 0
    for node_index in range(n_nodes):
        shards_sum_current_node = np.zeros(n_classes, dtype=int)
        for shard_per_node in range(n_shards_per_node):
            shards_sum_current_node += shards_info[
                shard_counter + shard_per_node]
        shard_counter += n_shards_per_node
        nodes_distribution.append(shards_sum_current_node)

    return np.array(nodes_distribution)
