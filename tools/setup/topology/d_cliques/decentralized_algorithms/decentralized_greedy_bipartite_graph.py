import numpy as np

import greedy_with_pre_comp_of_D_changed as greedyFuncs
import functions
import decentralized_greedy_resolving_conflicts as dgrc


class DecentralizedClique(greedyFuncs.Clique):
    def __init__(self, node_distribution, approximate_global_distribution,
                 node_id):
        super().__init__(node_distribution)
        self.approximate_global_distribution = approximate_global_distribution
        self.sum_all_nodes_approximate = approximate_global_distribution
        self.node_id = node_id
        self.nodes_that_prefer_me = []
        self.current_preferee = None
        self.will_live = False
        self.group = None  # 0: choose from 1 group, 1 choose
        # the best 0 if there are more than a 0 that prefers him
        self.nodes_ids = [node_id]

    def get_group(self):
        return self.group

    def get_nodes_that_prefer_me(self):
        return self.nodes_that_prefer_me

    def reset_node(self, preferee, node_id):
        self.current_preferee = preferee
        self.node_id = node_id
        self.will_live = False
        self.nodes_that_prefer_me = []

    def merge_clique(self, clique):
        self.will_live = True
        for i in range(len(clique.all_nodes)):
            self.add_node(clique.all_nodes[i])
        self.sum_all_nodes_approximate += clique.sum_all_nodes_approximate
        self.approximate_global_distribution = \
            self.sum_all_nodes_approximate / len(clique.all_nodes)
        self.nodes_ids.extend(clique.nodes_idsZ)

    def choose_group(self, rng):
        self.group = rng.integers(2)

    def first_communication_round(self, all_cliques):
        if self.current_preferee is None:
            self.will_live = True
            return
        # We ensure the preferee node is from group 1, since when we reset
        # the node we only pass neighbours from group 1.
        all_cliques[self.current_preferee].nodes_that_prefer_me.append(
            self.node_id)

    def second_communication_round(self, all_cliques):
        if not self.nodes_that_prefer_me:
            self.will_live = True
            return
        if len(self.nodes_that_prefer_me) == 1:
            # and all_cliques[self.nodes_that_prefer_me[-1]].get_group()== 0:
            self.merge_clique(all_cliques[self.nodes_that_prefer_me[0]])
            return

        cliques_skew = []
        # We ensure that nodes from group 1 will only have nodes from group 0
        # that prefer them. Because, we only perform first round of
        # communication on nodes from group 0
        if self.nodes_that_prefer_me:
            for node_that_prefer_me in self.nodes_that_prefer_me:
                cliques_skew.append(
                    greedyFuncs.skew(all_cliques[node_that_prefer_me],
                                     self.approximate_global_distribution))

            best_skew_index = np.argmin(np.array(cliques_skew))
            self.merge_clique(
                all_cliques[self.nodes_that_prefer_me[best_skew_index]])
            for node_that_prefer_me_index in range(
                    len(self.nodes_that_prefer_me)):
                if node_that_prefer_me_index != best_skew_index:
                    all_cliques[self.nodes_that_prefer_me[
                        node_that_prefer_me_index]].will_live = True


def decentralized_greedy_bipartite_solution(all_cliques, rng,
                                            global_distribution,
                                            number_of_cliques=None,
                                            size_of_cliques=None,
                                            directed="",
                                            random_sample=10,
                                            iterations=10,
                                            max_n_nodes=None):
    """
    :param all_cliques: list of all nodes in the distributed system
    :param rng: random generator
    :param global_distribution:
    :param number_of_cliques: statistic
    :param size_of_cliques: statistic
    :param directed: graph type
    :param random_sample: number of neighbours of a node
    :param iterations: number of times the algorithm will be executed
    :param max_n_nodes: constraint on maximum number of nodes per clique
    :return: tuple with the distributed cliques and a list of average skew
    over iterations
    """
    average_skew_iterations = []
    number_of_nodes = -1  # To avoid possibility of getting undefined variable
    for _ in range(iterations):
        number_of_nodes = len(all_cliques)
        if number_of_cliques is not None:
            number_of_cliques.append(number_of_nodes)
        if number_of_nodes == 1:
            if size_of_cliques is not None:
                size_of_cliques.append([len(all_cliques[0].all_nodes)] * 3 +
                                       [0])
            break
        if size_of_cliques is not None:
            size_of_cliques.append(dgrc.get_sizes_of_cliques(all_cliques))

        cliques_group_zero = []
        cliques_group_one = []
        graph = functions.build_graph(number_of_nodes, f"RANDOM{directed}",
                                      min(random_sample, number_of_nodes - 1),
                                      rng)
        for i in range(number_of_nodes):
            all_cliques[i].choose_group(rng)  # group 0 or 1

            if all_cliques[i].get_group() == 0:
                cliques_group_zero.append(i)
            else:
                cliques_group_one.append(i)

        for i in range(number_of_nodes):
            neighbours_group_one = []  # Nodes only from group 1
            for clique_neighbour_index in graph[i]:
                if all_cliques[clique_neighbour_index].get_group() == 1:
                    neighbours_group_one.append(clique_neighbour_index)
            if neighbours_group_one:
                all_cliques[i].reset_node(
                    dgrc.choose_preferred_clique_greedily(i,
                                                          neighbours_group_one,
                                                          all_cliques,
                                                          max_n_nodes), i)
            else:
                all_cliques[i].reset_node(None, i)

        for g_zero_node_index in cliques_group_zero:
            all_cliques[g_zero_node_index].first_communication_round(
                all_cliques)
        for g_one_node_index in cliques_group_one:
            all_cliques[g_one_node_index].second_communication_round(
                all_cliques)

        new_cliques = []
        for i in range(number_of_nodes):
            if all_cliques[i].will_live:
                new_cliques.append(all_cliques[i])
        all_cliques = new_cliques

        average_skew, max_skew, min_skew, std = \
            greedyFuncs.get_statistics_skew_from_distributed_cliques(
                all_cliques,
                global_distribution
            )
        average_skew_iterations.append([min_skew, average_skew, max_skew, std])
        print(f"Len of cliques: {len(all_cliques)}")
    if size_of_cliques is not None:
        size_of_cliques.append(dgrc.get_sizes_of_cliques(all_cliques))
    if number_of_cliques is not None:
        number_of_cliques.append(number_of_nodes)
    return all_cliques, average_skew_iterations
