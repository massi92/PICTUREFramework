import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import sys

from Framework.Cluster.Cluster import Cluster
from Framework.Snapshot.Snapshot import Snapshot
from Framework.DatasetReader.DatasetReader import DatasetReader


# VECCHIA VERSIONE DI REBUSCHI
# lasciata dentro per ricordo... nel caso potesse tornare utile


def find_closest_cluster(point: list, list_cluster: list) -> tuple:
    """
    This function given a point find the nearest cluster and compute the distance
    :param point:
    :param list_cluster:
    :return:
    """
    min_distance = sys.float_info.max
    closest_cluster = None
    for cluster in list_cluster:
        np_centroid = np.asarray(cluster.centroid)
        np_point = np.asarray(point)
        distance = np.linalg.norm(np_centroid - np_point)
        if distance < min_distance:
            closest_cluster = cluster
            min_distance = distance
    return closest_cluster, min_distance


def get_densest_cluster(list_cluster: list, dimension: int):
    """
    This function given a list of cluster get the densest cluster.
    :param list_cluster:
    :param dimension:
    :return:
    """
    if len(list_cluster) == 1:
        return list_cluster[0]
    max_density = 0.0
    densest_cluster = None
    for cluster in list_cluster:
        if float(cluster.radius) != 0.0:
            density = cluster.nb_points / math.pow(cluster.radius, dimension)
        else:
            density = 100
        if density > max_density:
            max_density = density
            densest_cluster = cluster
    return densest_cluster


def get_best_distance_cluster_to_cluster(cluster_selected: Cluster, list_cluster: list):
    min_distance = sys.float_info.max
    for cluster in list_cluster:
        if cluster != cluster_selected:
            distance = np.linalg.norm(cluster.centroid - cluster_selected.centroid)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def get_oldest_cluster(list_cluster: list, threshold: int) -> Cluster:
    """
    This function given a list of cluster get the oldest cluster
    :param list_cluster:
    :param threshold:
    :return:
    """
    min_relevance_timestamp = sys.float_info.max
    oldest_cluster = None
    for cluster in list_cluster:
        relevance_timestamp = cluster.get_relevance_timestamp(m=10)
        if relevance_timestamp < threshold and relevance_timestamp < min_relevance_timestamp:
            min_relevance_timestamp = relevance_timestamp
            oldest_cluster = cluster
    return oldest_cluster


def get_max_identifier(list_cluster: list, list_cluster_rem: list) -> int:
    """
    This function given a list cluster find the max id value and add one to the value
    :param list_cluster:
    :param list_cluster_rem:
    :return:
    """
    list_index = []
    list_index.extend(get_list_id_cluster(list_cluster))
    list_index.extend(get_list_id_cluster(list_cluster_rem))
    return max(list_index) + 1


def get_list_id_cluster(list_cluster: list):
    """
    This function given a list of cluster extract the list of id
    :param list_cluster:
    :return:
    """
    list_id = []
    for cluster in list_cluster:
        list_id.append(cluster.identifier)
    return list_id


def get_cluster_under_treshold(list_cluster: list, treshold: int) -> tuple:
    """
    This function given a list of cluster get clusters that have age under threshold. The age of
    a cluster is get by the get relevance timestamp function.
    :param list_cluster:
    :param treshold:
    :return:
    """
    oldest_cluster = []
    for cluster in list_cluster:
        relevance_timestamp = cluster.get_relevance_timestamp(m=10)
        if relevance_timestamp < treshold:
            oldest_cluster.append(cluster)
    if len(oldest_cluster) > 0:
        return True, oldest_cluster
    return False, oldest_cluster


def merge_two_clusters(cluster1: Cluster, cluster2: Cluster, list_cluster: list, list_cluster_removed: list) -> Cluster:
    """
    This function implements merge operation given two cluster
    """
    total_nb_points = cluster1.nb_points + cluster2.nb_points
    total_linear_sum = cluster1.linear_sum + cluster2.linear_sum
    total_squared_sum = cluster1.squared_sum + cluster2.squared_sum
    total_linear_time_sum = cluster1.linear_time_sum + cluster2.linear_time_sum
    total_squared_time_sum = cluster1.squared_time_sum + cluster2.squared_time_sum
    id_new_cluster = get_max_identifier(list_cluster, list_cluster_removed)
    new_cluster = Cluster(id_new_cluster, total_nb_points, [], [], total_linear_sum, total_squared_sum,
                          total_linear_time_sum, total_squared_time_sum, cluster1.centroid, 0.0)
    new_cluster.centroid = new_cluster.calculate_centroid()
    new_cluster.radius = new_cluster.calculate_radius()
    if not len(cluster1.id_list_sons) != 0:
        if not len(cluster2.id_list_sons) != 0:
            new_cluster.id_list_sons.extend([cluster1.identifier, cluster2.identifier])
        else:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.append(cluster1.identifier)
            new_cluster.id_list_sons.extend(cluster2.id_list_sons)
    else:
        if len(cluster2.id_list_sons) != 0:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.extend(cluster1.id_list_sons)
            new_cluster.id_list_sons.append(cluster2.identifier)
        else:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.extend(cluster1.id_list_sons)
            new_cluster.id_list_sons.extend(cluster2.id_list_sons)
    list_cluster.remove(cluster1)
    list_cluster.remove(cluster2)
    list_cluster_removed.extend([cluster1, cluster2])
    list_cluster.append(new_cluster)
    return new_cluster


class ClustreamDensity:

    def __init__(self,
                 dataset_reader: DatasetReader,
                 number_of_clusters=3,
                 time_window_length=40,
                 p=0.2,
                 threshold=8000):
        self.map = {"x": 0, "y": 1, "z": 2}
        self.number_of_clusters = number_of_clusters
        self.time_window_length = time_window_length
        self.p = p
        self.threshold = threshold
        self.dataset = dataset_reader.get_dataset()
        self.dataset_reader = dataset_reader
        self.rows_id = self.dataset.index
        self.dimension = self.dataset.keys()
        self.snapshot = None
        self.initialize_clusters()
        self.dataset = self.dataset.iloc[number_of_clusters:]
        self.list_snapshot = []
        self.extra_desc = self.get_string_parameters()

        self.finished = False

    def get_string_parameters(self):
        return 'Threshold: ' + str(self.threshold) + ', p: ' + str(self.p) + ', N Clusters: ' + str(
            self.number_of_clusters) + ', window: ' + str(self.time_window_length)

    def get_name(self):
        return 'Clustream-Density'

    def plot_clusters(self, axis, show_snapshots = False):
        columns = ['timestamp', 'centroids']
        columns_values = []
        for snapshot in tqdm(self.snapshots, desc='Plotting Clustream Clusters'):
            time = snapshot.timestamp
            for cluster in snapshot.clusters:
                columns_values.append([time, float(cluster.centroid_coordinates)])
                if show_snapshots:
                    axis.scatter(time, cluster.centroid_coordinates, c='red', s=15, alpha=1)
        return pd.DataFrame(data=columns_values, columns=columns)

    def get_snapshots(self) -> list[Snapshot]:
        if self.finished:
            return self.list_snapshot
        return []

    @property
    def snapshots(self):
        if self.finished:
            return self.list_snapshot
        return []

    def get_centroid_list(self):
        matrix_centroid = []
        list_timestamp = []
        for snapshot in tqdm(self.list_snapshot, desc="Computing Clustream Centroids"):
            list_centroid = []
            list_timestamp.append(snapshot.timestamp)
            for synthesis in snapshot.list_cluster:
                centroid = synthesis.centroid[0]
                list_centroid.append(centroid)
            matrix_centroid.append(list_centroid)
        return matrix_centroid, list_timestamp

    def initialize_clusters(self):
        i = 0
        list_cluster = []
        list_data = {}
        for index, row in self.dataset.iloc[0: self.number_of_clusters].iterrows():
            index = math.ceil(index)
            square_values = np.sum(np.square(row.values))
            square_window = np.square(self.time_window_length)
            cluster = Cluster(i, 1, [], [], row.values, square_values, index, square_window, row.values, 0.0)
            list_cluster.append(cluster)
            list_data[cluster.identifier] = []
            list_data[cluster.identifier].append(row.values)
            i += 1
        snapshot = Snapshot()
        snapshot.timestamp = 0
        snapshot.dimension = self.dimension
        snapshot.list_cluster = list_cluster
        snapshot.list_cluster_removed = []
        snapshot.list_original_data = list_data
        snapshot.list_original_data_removed = {}
        self.snapshot = snapshot

    def remove_cluster(self, list_cluster: list, list_removed: list, treshold: int, dimension: int) -> tuple:
        """
        This function implements the remove operation for clustream-d, the remove
        operation is based on the age and on the density
        """
        result, oldest_cluster = get_cluster_under_treshold(list_cluster, treshold)
        if not result:
            return False, None
        densest_cluster = get_densest_cluster(oldest_cluster, dimension)
        for cluster in oldest_cluster:
            if cluster.identifier == densest_cluster.identifier:
                list_cluster.remove(cluster)
                list_removed.append(cluster)
                return True, densest_cluster
        return False, None

    def run(self):
        dataset = self.dataset.to_dict('records')
        snapshot = self.snapshot
        self.list_snapshot = []
        i = 0
        for row in tqdm(dataset, desc="Executing Clustream | " + self.extra_desc):
            row_data = np.asarray(list(row.values()))
            index = self.rows_id[i]
            i = i + 1
            snapshot = self.update_clusters(snapshot, row_data, index)
            if index != self.number_of_clusters:
                if ((index - self.number_of_clusters) % self.time_window_length) == 0 or index == self.rows_id[-1]:
                    self.list_snapshot.append(copy.deepcopy(snapshot))
                    # if save_snap:
                    # save_csv_snapshot(snapshot_density, snapshot_no_density, path_snapshot, index, string_result)
                    self.snapshot.list_cluster_removed = []
                    self.snapshot.list_original_data_removed = {}
        self.finished = True

    def update_clusters(self,
                        snapshot: Snapshot,
                        new_point: np.array,
                        timestamp: int):
        temp_snapshot = snapshot
        temp_snapshot.timestamp = timestamp
        closest_cluster, result = self.get_operation(new_point, temp_snapshot.list_cluster)
        if result == "ABSORB":
            closest_cluster.insert_point_cluster(new_point, timestamp)
            temp_snapshot.list_original_data[closest_cluster.identifier].append(new_point)
        elif result == "CREATE":
            result_remove, cluster_to_remove = self.remove_cluster(temp_snapshot.list_cluster,
                                                                   temp_snapshot.list_cluster_removed,
                                                                   self.threshold, new_point.size)
            if result_remove:
                temp_snapshot.list_original_data_removed[
                    cluster_to_remove.identifier] = temp_snapshot.list_original_data
                del temp_snapshot.list_original_data[cluster_to_remove.identifier]
            if not result_remove:
                min_distance = sys.float_info.max
                cluster1 = None
                cluster2 = None
                for i, cluster in enumerate(temp_snapshot.list_cluster):
                    center = cluster.centroid
                    for next_cluster in temp_snapshot.list_cluster[i + 1:]:
                        dist = np.linalg.norm(center - next_cluster.centroid)
                        if dist < min_distance:
                            min_distance = dist
                            cluster1 = cluster
                            cluster2 = next_cluster
                new_cluster = merge_two_clusters(cluster1, cluster2, temp_snapshot.list_cluster,
                                                 temp_snapshot.list_cluster_removed)
                temporal_data = []
                for data in temp_snapshot.list_original_data[cluster1.identifier]:
                    temporal_data.append(data)
                for data in temp_snapshot.list_original_data[cluster2.identifier]:
                    temporal_data.append(data)
                del temp_snapshot.list_original_data[cluster1.identifier]
                del temp_snapshot.list_original_data[cluster2.identifier]
                temp_snapshot.list_original_data[new_cluster.identifier] = temporal_data

            new_index = get_max_identifier(temp_snapshot.list_cluster, temp_snapshot.list_cluster_removed)
            squared_values = np.sum(np.power(new_point, 2))
            new_cluster = Cluster(new_index, 1, [], [], new_point, squared_values, timestamp,
                                  math.pow(timestamp, 2), new_point, 0.0)
            temp_snapshot.list_cluster.append(new_cluster)

            temp_snapshot.list_original_data[new_index] = []
            temp_snapshot.list_original_data[new_index].append(new_point)
        return temp_snapshot

    def get_operation(self, point: list, list_cluster: list) -> tuple:
        closest_cluster, distance = find_closest_cluster(point, list_cluster)
        if closest_cluster.nb_points > 1:
            maximal_boundary = self.p * closest_cluster.calculate_radius()
        else:
            maximal_boundary = get_best_distance_cluster_to_cluster(closest_cluster, list_cluster)
        if distance < maximal_boundary:
            return closest_cluster, "ABSORB"
        else:
            return closest_cluster, "CREATE"
