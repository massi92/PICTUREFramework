import warnings

import pandas as pd

from Framework.Algorithms.Algorithm import Algorithm
from Framework.Algorithms.Dstream.DStreamCharacteristicVector import DStreamCharacteristicVector
from Framework.Algorithms.Dstream.Snapshot import DStreamSnapshot

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import random

from tqdm import tqdm

from Framework.DatasetReader.DatasetReader import DatasetReader

import networkx as nx
from Framework.Algorithms.Dstream.dstream_util import get_sporadic_grids, get_last_label_changed, get_dense_grids, \
    get_grids_of_cluster_class, generate_unique_class_key, reset_last_label_changed, \
    get_most_recently_categorically_changed_grids


class DStream(Algorithm):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 dense_threshold_parameter=1.8,  # 3.0, #C_m
                 sparse_threshold_parameter=1.7,  # 0.8,  #C_l
                 sporadic_threshold_parameter=0.3,  # 0.3, #beta
                 decay_factor=0.997,  # 0.998, #lambda
                 dimensions=2,
                 domains_per_dimension=((0.0, 100.0), (0.0, 100.0)),
                 partitions_per_dimension=(5, 5),
                 initial_cluster_count=1,
                 seed=331,
                 window=10):
        np.seterr(divide='ignore', invalid='ignore')

        self.dense_threshold_parameter = dense_threshold_parameter
        self.sparse_thresold_parameter = sparse_threshold_parameter
        self.sporadic_threshold_parameter = sporadic_threshold_parameter
        self.decay_factor = decay_factor

        self.dimensions = dimensions
        self.domains_per_dimension = domains_per_dimension
        self.partitions_per_dimension = partitions_per_dimension

        self.snapshots = []

        N = 1
        for i in range(dimensions):
            N *= partitions_per_dimension[i]

        self.maximum_grid_count = N
        self.grids = {}
        self.removed_grid_cache = {}
        self.cluster_count = initial_cluster_count

        self.initial_cluster_count = initial_cluster_count

        self.gap_time = -1.0
        self.compute_gap_time()
        # calcolo gap time -> funzione rimossa per pulizia
        quotient1 = self.sparse_thresold_parameter / self.dense_threshold_parameter
        quotient2 = (self.maximum_grid_count - self.dense_threshold_parameter) / (
                self.maximum_grid_count - self.sparse_thresold_parameter)
        max_val = np.max([quotient1, quotient2])
        max_log = np.log(max_val) / np.log(self.decay_factor)
        gap = np.floor([max_log])
        self.gap_time = gap[0]

        self.has_clustered_once = False

        self.current_time = 0

        self.seed = seed
        self.class_keys = np.array([])
        random.seed(self.seed)

        self.data = np.array([])

        self.dataset = dataset_reader.get_dataset()
        self.dataset_reader = dataset_reader

        self.dense_param = self.dense_threshold_parameter / (self.maximum_grid_count * (1.0 - self.decay_factor))
        self.sparse_param = self.sparse_thresold_parameter / (self.maximum_grid_count * (1.0 - self.decay_factor))

        self.window = window
        self.windowed = False

    def reset_parameters(self):
        self.grids = {}
        self.removed_grid_cache = {}
        self.cluster_count = self.initial_cluster_count
        self.has_clustered_once = False
        self.class_keys = np.array([])

    def get_snapshots(self) -> list[DStreamSnapshot]:
        return self.snapshots

    # Comodo per fare i titoli dei grafici
    def get_string_parameters(self):
        return 'Dense Threshold Parameter: ' + str(self.dense_threshold_parameter) + ', Sparse Threshold Parameter: ' \
            + str(self.sparse_thresold_parameter) + ', Sporadic Threshold Parameter: ' + \
            str(self.sporadic_threshold_parameter) + ', Decay Factor: ' + str(self.decay_factor) + ', Window: ' + str(
                self.window)

    def get_name(self):
        return 'D-Stream'


    def plot_clusters(self, axis, show_snaps=False):
        partitions = self.partitions_per_dimension[0]
        top = self.domains_per_dimension[0][1]
        bottom = self.domains_per_dimension[0][0]
        columns = ['timestamp', 'centroids', 'type']
        columns_values = []
        size = 10
        for s in tqdm(self.snapshots, desc='Plotting DStream Clusters'):
            stamp = s.timestamp
            snapshot = s.clusters
            for x in snapshot:
                if x.density_category == 'DENSE':
                    if show_snaps:
                        axis.scatter(stamp, x.centroid_coordinates, c='black', s=size)
                    columns_values.append([stamp, float(x.centroid_coordinates), 'DENSE'])
                elif x.density_category == 'SPARSE':
                    if show_snaps:
                        axis.scatter(stamp, x.centroid_coordinates, c='red', s=size)  # was orange
                    columns_values.append([stamp, float(x.centroid_coordinates), 'SPARSE'])
                elif x.density_category == 'TRANSITIONAL':
                    if show_snaps:
                        axis.scatter(stamp, x.centroid_coordinates, c='yellow', s=size)
                    columns_values.append([stamp, float(x.centroid_coordinates), 'TRANSITIONAL'])
                else:
                    if show_snaps:
                        axis.scatter(stamp, x.centroid_coordinates, c='red', s=size)
                    columns_values.append([stamp, float(x.centroid_coordinates), 'NONE'])
        return pd.DataFrame(data=columns_values, columns=columns)

    def compute_gap_time(self):
        # Vedi paper di DStream per capire come mai il gap time viene calcolato in questa maniera
        quotient1 = self.sparse_thresold_parameter / self.dense_threshold_parameter
        quotient2 = (self.maximum_grid_count - self.dense_threshold_parameter) / (
                self.maximum_grid_count - self.sparse_thresold_parameter)
        max_val = np.max([quotient1, quotient2])
        max_log = np.log(max_val) / np.log(self.decay_factor)
        gap = np.floor([max_log])
        self.gap_time = gap[0]
        # print('gap params: ', quotient1, quotient2, max_val, max_log, gap)
        # print('computed gap time: ', self.gap_time)

    def run(self, column="number_sold"):
        # metodo che esegue l'algoritmo...
        data = np.array(self.dataset[column])
        i = 0
        self.windowed = False
        indexes = []
        for d in tqdm(data, desc="Running DStream"):
            datum = (int(d), i)
            i += 1
            self.add_datum(datum)
            # print(datum)
            indexes.append(self.get_grid_indices(datum))

    def windowed_run(self, column='number_sold'):
        # metodo che esegue l'algoritmo con il finestramento landmark
        data = np.array(self.dataset[column])
        i = 0
        self.windowed = True
        indexes = []
        for d in tqdm(data, desc="Running DStream Windowed"):
            datum = (int(d), i)
            i += 1
            indexes.append(self.get_grid_indices(datum))
            if i % self.window == 0:
                self.add_datum(datum, True)
                self.reset_parameters()
            else:
                self.add_datum(datum, False)

    def get_cluster_centers(self):
        x = []
        y = []
        if self.dimensions == 1:
            for key in self.grids:
                # magari prendo solamente quelle dense?
                x.append(key[0] * (
                        (self.domains_per_dimension[0][1] - self.domains_per_dimension[0][0])
                        / self.partitions_per_dimension[0]
                ))
                return x
        elif self.dimensions == 2:
            for key in self.grids:
                x.append(key[0] * (
                        (self.domains_per_dimension[0][1] - self.domains_per_dimension[0][0])
                        / self.partitions_per_dimension[0]
                ))
                y.append(key[1] * (
                        (self.domains_per_dimension[1][1] - self.domains_per_dimension[1][0])
                        / self.partitions_per_dimension[1]
                ))
                return x, y
        return None

    def get_grid_indices(self, datum):
        # funzione concettualmente semplicissima
        # fa il rapporto fra i limiti del dominio e fornisce
        # l'indice corrispettivo alla casella
        # OCCHIO: l'indice non è la coordinata, quando si vogliono plottare
        # i centroidi bisogna fare la proporzione di nuovo (già fatto, la conversione viene effettuata
        # automaticamente quando le celle vengono inserite in uno Snapshot, vedere la corrispettiva classe
        # (funzione convert_indexes))
        indices = np.array([])
        for i in range(self.dimensions):
            domain_tuple = self.domains_per_dimension[i]
            partitions = self.partitions_per_dimension[i]
            domain_size = domain_tuple[1] - domain_tuple[0]
            test_datum = datum[i] - domain_tuple[0]
            index = np.floor([(test_datum / domain_size) * (partitions)])[0]
            if index >= partitions:
                # print('index equals partitions: ', index, partitions)
                index = partitions - 1
            indices = np.append(indices, index)
        return indices

    def density_threshold_function(self, last_update_time, current_time):
        # paper eq 27
        top = self.sparse_thresold_parameter * (1.0 - self.decay_factor ** (current_time - last_update_time + 1))
        bottom = self.maximum_grid_count * (1.0 - self.decay_factor)
        return top / bottom

    def update_density_category(self):
        for indices, grid in self.grids.items():
            dense_param = self.dense_param
            sparse_param = self.sparse_param
            test_density = grid.density
            if test_density >= dense_param:
                if grid.density_category != 'DENSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False

                grid.density_category = 'DENSE'
            if test_density <= sparse_param:
                if grid.density_category != 'SPARSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'SPARSE'
            if test_density >= sparse_param and grid.density <= dense_param:
                if grid.density_category != 'TRANSITIONAL':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'TRANSITIONAL'
            self.grids[indices] = grid

    def are_neighbors(self, grid1_indices, grid2_indices):
        # se interessa solo la parte online ignorare...
        target_identical_count = self.dimensions - 1
        identical_count = 0
        for i in range(self.dimensions):
            if grid1_indices[i] == grid2_indices[i]:
                identical_count += 1
            elif np.abs(grid1_indices[i] - grid2_indices[i]) != 1:
                return False
        return identical_count == target_identical_count

    def get_graph_of_cluster(self, grids):
        # se interessa solo la parte online ignorare...
        indices_list = grids.keys()
        g = nx.empty_graph()
        for i in range(len(indices_list)):
            indices = list(indices_list)[i]
            for j in range(len(indices_list)):
                other_indices = list(indices_list)[j]
                if self.are_neighbors(indices, other_indices):
                    if g.has_edge(indices, other_indices) == False:
                        g.add_edge(indices, other_indices)
                        continue
                g.add_node(other_indices)
            if g.has_node(indices) == False:
                g.add_node(indices)
        return g

    def get_neighboring_grids(self, ref_indices, cluster_grids=None):
        # se interessa solo la parte online ignorare...
        if cluster_grids != None:
            to_get_neighbors_amongst = cluster_grids
        else:
            to_get_neighbors_amongst = self.grids
        to_get_neighbors_amongst[ref_indices] = self.grids[ref_indices]
        to_get_neighbors_amongst_graph = self.get_graph_of_cluster(to_get_neighbors_amongst)
        neighbors = to_get_neighbors_amongst_graph.neighbors(ref_indices)
        neighbors_dict = {}
        neighbors_list = list(neighbors)
        for i in range(len(list(neighbors))):
            indices = neighbors_list[i]
            neighbors_dict[indices] = self.grids[indices]
        if len(neighbors_dict) == 0:
            return None
        return neighbors_dict

    def get_inside_grids(self, grids):
        # se interessa solo la parte online ignorare...
        inside_grids = {}
        outside_grids = {}
        target_inside_neighbor_count = 2 * self.dimensions
        for indices, grid in grids.items():
            neighboring_grids = self.get_neighboring_grids(indices, grids)
            if neighboring_grids != None:
                if len(neighboring_grids.keys()) == target_inside_neighbor_count:
                    inside_grids[indices] = grid
                else:
                    outside_grids[indices] = grid

        return inside_grids, outside_grids

    def is_valid_cluster(self, grids):
        for indices, grid in grids.items():
            for indices2, grid2 in grids.items():
                if indices != indices2:
                    if not self.are_neighbors(indices, indices2):
                        return False

        inside_grids, outside_grids = self.get_inside_grids(grids)

        for indices, grid in inside_grids.items():
            if grid.density_category != 'DENSE':
                return False
        for indices, grid in outside_grids.items():
            if grid.density_category == 'DENSE':
                return False

        return True

    def assign_to_cluster_class(self, grids, class_key):
        for indices, grid in grids.items():
            if grid.label != class_key:
                grid.label_changed_last_iteration = True
            grid.label = class_key

            self.grids[indices] = grid

    def validate_can_belong_to_cluster(self, cluster, test_grid):
        is_valid_before = self.is_valid_cluster(cluster)
        if is_valid_before != True:
            return False

        cluster[test_grid[0]] = test_grid[1]
        is_valid_after = self.is_valid_cluster(cluster)
        return is_valid_after

    def initialize_clusters(self):
        self.update_density_category()
        cluster_counts = np.array([])
        dense_grids, non_dense_grids = get_dense_grids(self.grids)

        if len(dense_grids.keys()) < self.cluster_count:
            # print('not enough dense clusters')
            # self.cluster_count = 0
            return

        cluster_size = np.round(len(dense_grids.keys()) / self.cluster_count)

        # print('cluster size: ', cluster_size)
        for i in range(self.cluster_count):
            if i == self.cluster_count - 1:
                current_total = np.sum(cluster_counts)
                last_count = len(dense_grids.keys()) - current_total
                cluster_counts = np.append(cluster_counts, int(last_count))
                # print('last cluster size: ', last_count)
            else:
                cluster_counts = np.append(cluster_counts, int(cluster_size))
        counter = 0
        # print(cluster_counts)
        for grid_count in cluster_counts:
            grid_count = int(grid_count)
            cluster_grids = {}
            unique_class_key = generate_unique_class_key(self.class_keys)  # genera una chiave casuale
            self.class_keys = np.append(self.class_keys, unique_class_key)
            # print(grid_count)
            for i in range(grid_count):
                k = list(dense_grids.keys())[counter]
                v = list(dense_grids.values())[counter]
                v.label = unique_class_key
                cluster_grids[k] = v
                counter += 1
        for indices, grid in non_dense_grids.items():
            grid.label = 'NO_CLASS'
            self.grids[indices] = grid

        iter_count = 0
        last_label_changed_grids = get_last_label_changed(self.grids)
        last_label_changed_grids_2 = get_last_label_changed(self.grids)
        diff = np.setdiff1d(np.array(last_label_changed_grids.keys()), np.array(last_label_changed_grids_2.keys()))

        while iter_count == 0 or diff.size > 0:  # last_label_changed_grids.keys()#len(last_label_changed_grids.keys()) != 0:
            # print('iter_count: ', iter_count)
            # raw_input('waiting on return')
            # print(last_label_changed_grids.keys(), (self.grids[list(last_label_changed_grids.keys())[0]]).label)
            iter_count += 1
            for i in range(self.class_keys.size):

                class_key = self.class_keys[i]
                # print('class_key: ', class_key)
                cluster_grids = get_grids_of_cluster_class(self.grids, class_key)
                inside_grids, outside_grids = self.get_inside_grids(cluster_grids)
                # print('inside grid count: {} outside grid count: {} total grid_count: {}'.format(
                #     inside_grids.keys().__len__(), outside_grids.keys().__len__(), self.grids.keys().__len__()))
                for indices, grid in outside_grids.items():
                    neighboring_grids = self.get_neighboring_grids(indices)
                    for neighbor_indices, neighbor_grid in neighboring_grids.items():
                        # print('class key sizes: ', self.class_keys.size)
                        for j in range(self.class_keys.size):
                            # print(j)
                            test_class_key = self.class_keys[j]
                            test_cluster_grids = get_grids_of_cluster_class(self.grids, test_class_key)

                            neighbor_belongs_to_test_cluster = self.validate_can_belong_to_cluster(test_cluster_grids, (
                                neighbor_indices, neighbor_grid))
                            reset_last_label_changed(self.grids)
                            if neighbor_belongs_to_test_cluster:
                                if len(cluster_grids.keys()) > len(test_cluster_grids.keys()):
                                    self.assign_to_cluster_class(test_cluster_grids, class_key)
                                else:
                                    self.assign_to_cluster_class(cluster_grids, test_class_key)
                            elif neighbor_grid.density_category == 'TRANSITIONAL':
                                self.assign_to_cluster_class({neighbor_indices: neighbor_grid}, class_key)
                                # self.update_class_keys()
            last_label_changed_grids_2 = last_label_changed_grids
            last_label_changed_grids = get_last_label_changed(self.grids)
            diff = np.setdiff1d(np.array(last_label_changed_grids.keys()), np.array(last_label_changed_grids_2.keys()))
        self.has_clustered_once = True

    def is_sporadic(self, grid, current_time):
        if grid.density < self.density_threshold_function(grid.last_update_time, current_time) and current_time >= (
                1.0 + self.sporadic_threshold_parameter) * grid.last_sporadic_removal_time:
            return True
        return False

    def detect_sporadic_grids(self, current_time):
        for indices, grid in self.grids.items():
            if self.is_sporadic(grid, current_time):
                grid.status = 'SPORADIC'
                grid.last_marked_sporadic_time = current_time
            else:
                grid.status = 'NORMAL'
            self.grids[indices] = grid

    def add_datum(self, datum, force_snap=False):  # METODO USATO ALL'INTERNO DI RUN
        if self.data.size == 0:  # AGGIORNAMENTO DEI MICROCLUSTERS
            self.data = np.array(datum)  # CORRISPONDE A TEMPLATE METHOD DI REBUSCHI? ~CIRCA
        else:
            self.data = np.row_stack((self.data, np.array(datum)))

        # Trovo la posizione del datum nella griglia
        indices = tuple(self.get_grid_indices(datum))

        # se l'indice trovato e' gia' stato inserito nella griglia
        # seleziono la casella grid da grids che e' la griglia grande
        if indices in self.grids:
            grid = self.grids[indices]
        else:
            # altrimenti se l'indice e' fra quelli rimossi
            if indices in self.removed_grid_cache:
                # seleziono la casella da quelli rimossi
                grid = self.removed_grid_cache[indices]
            else:
                grid = DStreamCharacteristicVector()
                # altrimenti creo una casella base
                # da zero

        # ------------------------------------------------- questa parte si può mettere dentro la classe?
        grid.update_density(self.decay_factor, self.current_time)
        # # calcolo la densita' aggiornata
        # grid.density = 1.0 + grid.density * self.decay_factor ** (self.current_time - grid.last_update_time)
        # # aggiorna il tempo di visita della casella
        # grid.last_update_time = self.current_time
        # # aggiorna la tabella generale aggiornando l'indice specifico

        grid.add_sample(datum)

        self.grids[indices] = grid

        # eventuale clustering iniziale
        if self.current_time >= self.gap_time and not self.has_clustered_once:
            self.initialize_clusters()

        # cluster effettivo ogni gap time
        if np.mod(self.current_time, self.gap_time) == 0 and self.has_clustered_once:
            # time to save a snapshot
            # should create an object which contains only the gridboxes
            # that are 'normal'.. or one for everytyepe
            # grids è una semplice lista
            # ogni box è un cluster, e uno snapshot consiste nel salvare tutti i microcluster con la situazione del momento
            # non dovrebbe essere troppo difficile
            sporadic_grids = get_sporadic_grids(self.grids)

            for indices, grid in sporadic_grids.items():
                if grid.last_marked_sporadic_time != -1 and grid.last_marked_sporadic_time + 1 <= self.current_time:
                    if grid.last_update_time != self.current_time:

                        self.grids = {key: value for key, value in self.grids.items() if value is not grid}
                        # tutta sta menata di riga per dire che esclude la grid che non va bene
                        # questa sintassi python è sicuramente compatta... ma decisamente poco comprensibile...
                        grid.last_sporadic_removal_time = self.current_time
                        self.removed_grid_cache[indices] = grid

                    else:
                        if not self.is_sporadic(grid, self.current_time):
                            grid.status = 'NORMAL'
                            self.grids[indices] = grid
            self.detect_sporadic_grids(self.current_time)
            self.cluster()

        # SNAPSHOT SAVE -----------------------------------------------------------------------------------
        if (np.mod(self.current_time, self.window) == 0 and not self.windowed) or (force_snap and self.windowed):
            self.snapshots.append(
                DStreamSnapshot(self.grids, self.current_time, self.domains_per_dimension,
                                self.partitions_per_dimension))
        self.current_time += 1

    def cluster_still_connected_upon_removal(self, grids_without_removal, removal_grid):

        removal_grid_indices = removal_grid[0]
        # print 'removal grid indices: ', removal_grid_indices
        grids_with_removal = {key: value for key, value in grids_without_removal.items() if
                              key is not removal_grid_indices}

        # print 'connect grids with removal keys: ', grids_with_removal.keys()

        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        if graph_with_removal.size() == 0:
            return False
        return nx.is_connected(graph_with_removal)

    def extract_two_clusters_from_grids_having_just_removed_given_grid(self, grids_without_removal, removed_grid):
        # first remove it, then split into two, then add the two to self.grids
        removed_grid_indices = removed_grid[0]

        grids_with_removal = {key: value for key, value in grids_without_removal.items() if
                              key is not removed_grid_indices}
        # print 'ex2 grids with removal: ', grids_with_removal
        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        # subgraphs = nx.connected_component_subgraphs(graph_with_removal)
        subgraphs = (graph_with_removal.subgraph(c) for c in nx.connected_components(graph_with_removal))

        if len(list(subgraphs)) != 2:
            pass  # print 'found != 2 subgraphs; count: ', len(subgraphs)

        for i in range(len(list(subgraphs))):
            if i != 0:
                nodes = subgraphs[i].nodes()
                new_class_key = generate_unique_class_key(self.class_keys)
                self.class_keys = np.append(self.class_keys, new_class_key)
                for node in nodes:
                    grid = self.grids[node]
                    grid.label = new_class_key
                    self.grids[node] = grid

    def grid_becomes_outside_if_other_grid_added_to_cluster(self, test_grid, cluster_grids, insert_grid):
        cluster_grids[insert_grid[0]] = insert_grid[1]
        inside_grids, outside_grids = self.get_inside_grids(cluster_grids)
        if test_grid[0] in outside_grids:
            return True
        return False

    def grid_is_outside_if_added_to_cluster(self, test_grid, grids):
        grids = grids.copy()
        grids[test_grid[0]] = test_grid[1]
        inside_grids, outside_grids = self.get_inside_grids(grids)
        if outside_grids.has_key(test_grid[0]):
            return True
        return False

    def cluster(self):
        self.update_density_category()
        for indices, grid in get_most_recently_categorically_changed_grids(self.grids).items():

            neighboring_grids = self.get_neighboring_grids(indices)
            neighboring_clusters = {}
            if neighboring_grids != None:

                for neighbor_indices, neighbor_grid in neighboring_grids.items():
                    neighbors_cluster_class = neighbor_grid.label
                    neighbors_cluster_grids = get_grids_of_cluster_class(self.grids, neighbors_cluster_class)
                    neighboring_clusters[neighbor_indices, neighbors_cluster_class] = neighbors_cluster_grids
                if len(neighboring_grids.keys()) != 0:
                    max_neighbor_cluster_size = 0
                    max_size_indices = None
                    # max_size_cluster = None
                    # print 'neighboring clusters: ', neighboring_clusters.keys()
                    for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                        test_size = len(ref_neighbor_cluster_grids.keys())
                        # print 'size comparison: ', test_size, max_neighbor_cluster_size
                        if test_size > max_neighbor_cluster_size:
                            max_neighbor_cluster_size = test_size
                            # max_size_cluster = neighbor_cluster
                            max_size_cluster_key = k[1]
                            max_size_indices = k[0]
                            max_cluster_grids = ref_neighbor_cluster_grids
                    max_size_grid = neighboring_grids[max_size_indices]
                    grids_cluster = get_grids_of_cluster_class(self.grids, grid.label)

            if grid.density_category == 'SPARSE':
                changed_grid_cluster_class = grid.label
                cluster_grids_of_changed_grid = get_grids_of_cluster_class(self.grids, changed_grid_cluster_class)
                # print 'cluster grids of changed grid keys: ', cluster_grids_of_changed_grid.keys()
                would_still_be_connected = self.cluster_still_connected_upon_removal(cluster_grids_of_changed_grid,
                                                                                     (indices, grid))
                grid.label = 'NO_CLASS'
                self.grids[indices] = grid

                if would_still_be_connected == False:
                    self.extract_two_clusters_from_grids_having_just_removed_given_grid(cluster_grids_of_changed_grid,
                                                                                        (indices, grid))

            elif grid.density_category == 'DENSE':
                if len(neighboring_clusters.keys()) == 0:
                    # print('no neighbors, returning')
                    return
                if max_size_grid.density_category == 'DENSE':

                    if grid.label == 'NO_CLASS':
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) > max_neighbor_cluster_size:
                        if grid.label != 'NO_CLASS':
                            for max_indices, max_grid in max_cluster_grids.items():
                                max_grid.label = grid.label
                                self.grids[max_indices] = max_grid
                    elif len(grids_cluster.keys()) <= max_neighbor_cluster_size:
                        if max_size_cluster_key != 'NO_CLASS':
                            for grids_cluster_indices, grids_cluster_grid in grids_cluster.items():
                                grids_cluster_grid.label = max_size_cluster_key
                                self.grids[grids_cluster_indices] = grids_cluster_grid
                elif max_size_grid.density_category == 'TRANSITIONAL':
                    if grid.label == 'NO_CLASS' and self.grid_becomes_outside_if_other_grid_added_to_cluster(
                            (max_size_indices, max_size_grid), max_cluster_grids, (indices, grid)):
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) >= max_neighbor_cluster_size:
                        max_size_grid.label = grid.label
                        self.grids[max_size_indices] = max_size_grid
            elif grid.density_category == 'TRANSITIONAL':
                if len(neighboring_clusters.keys()) == 0:
                    # print('no neighbors, returning')
                    return
                max_outside_cluster_size = 0
                max_outside_cluster_class = None
                for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                    ref_cluster_key = k[1]
                    # ref_indices = k[0]
                    ref_grids = ref_neighbor_cluster_grids
                    if self.grid_is_outside_if_added_to_cluster((indices, grid), ref_grids) == True:
                        test_size = len(ref_grids.keys())
                        if test_size > max_outside_cluster_size:
                            max_outside_cluster_size = test_size
                            max_outside_cluster_class = ref_cluster_key
                grid.label = max_outside_cluster_class
                self.grids[indices] = grid

            self.update_class_keys()
        self.split_and_merge()

    def update_class_keys(self):
        new_keys = np.array([])
        # print 'updating class keys: ', self.class_keys
        for indices, grid in self.grids.items():
            if grid.label not in new_keys and grid.label != 'NO_CLASS' and grid.label is not None:
                # print 'new class key ', grid.label
                new_keys = np.append(new_keys, grid.label)
        self.class_keys = new_keys

    def split_and_merge(self):
        self.split()
        self.merge()
        self.update_class_keys()

    def split(self):
        for class_key in self.class_keys:
            if class_key == None or class_key == '' or class_key == []:
                # print class_key, ' is no good class key'
                continue
            # print 'attemping split of class {}'.format(class_key)
            cluster = get_grids_of_cluster_class(self.grids, class_key)
            cluster_graph = self.get_graph_of_cluster(cluster)
            # print 'cluster graph size {}'.format(len(cluster_graph.nodes()))

            if len(cluster_graph.nodes()) == 0:
                # for indices, grid in cluster.items():
                #     print(indices, grid.label)
                # print 'null graph, splitting each grid into cluster'
                # raw_input()
                for indices, grid in cluster.items():
                    self.create_new_cluster({indices: grid})
                continue
            # subgraphs = nx.connected_component_subgraphs(cluster_graph)
            subgraphs = [cluster_graph.subgraph(c) for c in nx.connected_components(cluster_graph)]
            if len(list(subgraphs)) != 1:

                # print('SPLIT', cluster.keys(), 'into {} clusters'.format(len(subgraphs)))
                for subgraph in subgraphs:
                    nodes = subgraph.nodes()
                    # print 'nodes: ', nodes
                    new_grids = {}
                    for node in nodes:
                        new_grids[node] = self.grids[node]
                    self.create_new_cluster(new_grids)
                # print 'class keys after: ', self.class_keys.size, self.class_keys
                self.update_class_keys()
                self.split()

    def merge(self):
        for class_key in self.class_keys:
            # print 'attempting merge of cluster class ', class_key
            cluster = get_grids_of_cluster_class(self.grids, class_key)
            if len(cluster.keys()) != 0:
                cluster_graph = self.get_graph_of_cluster(cluster)
                # print 'cluster graph size ', len(cluster_graph.nodes())
                for test_class_key in self.class_keys:
                    if test_class_key != class_key:
                        test_cluster = get_grids_of_cluster_class(self.grids, test_class_key)
                        if len(test_cluster.keys()) != 0:
                            if self.is_valid_cluster(dict(cluster.items() + test_cluster.items())) == False:
                                continue
                            test_cluster_graph = self.get_graph_of_cluster(test_cluster)

                            cg_copy = cluster_graph.copy()
                            # print 'adding {} of size {} to {} of size {}'.format(test_cluster_graph.nodes(), len(test_cluster_graph.nodes()), cg_copy.nodes(), len(cg_copy.nodes()))
                            cg_copy.add_edges_from(test_cluster_graph.edges())
                            if len(test_cluster_graph.nodes()) == 1:
                                cg_copy.add_node(test_cluster_graph.nodes()[0])
                            for node in cg_copy.nodes():

                                for test_node in cg_copy.nodes():
                                    if test_node != node:
                                        if self.are_neighbors(node, test_node):
                                            if cg_copy.has_edge(node, test_node) == False:
                                                # print 'adding edge ', (node, test_node)
                                                cg_copy.add_edge(node, test_node)

                            subgraphs = nx.connected_component_subgraphs(cg_copy)
                            if len(subgraphs) == 1:
                                # print('MERGE', cluster.keys(), test_cluster.keys())
                                if len(cluster.keys()) > len(test_cluster.keys()):
                                    self.assign_to_cluster_class(test_cluster, class_key)
                                else:
                                    self.assign_to_cluster_class(cluster, test_class_key)
                                self.update_class_keys()
                                # print 'after ', len(self.class_keys)
                                self.merge()
