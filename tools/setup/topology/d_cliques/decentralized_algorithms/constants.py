class Figures:
    COLOR_PRIMARY = "blue"
    COLOR_SECONDARY = "orange"
    COLOR_TERTIARY = "green"
    COLOR_QUATERNARY = "red"
    SAVE_PLOT = True


class ExperimentSetting:
    TESTS = 10
    N_NODES = 100
    N_CLASSES = 10
    SHARD_SIZE = 300
    SAMPLES_PER_CLASS = [6000] * 10
    DATA_PARTITION_MODE = "simple"  # Options: simple, shards


class Constraints:
    K = range(1, ExperimentSetting.N_NODES + 1)
    MAX_NODES_PER_CLIQUE_OPTIMAL = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    MAX_NODES_PER_CLIQUE = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
