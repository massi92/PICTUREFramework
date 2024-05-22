import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import sys

from Framework.Algorithms.Algorithm import Algorithm
from Framework.Algorithms.CluStream.clustream_util import get_oldest_cluster, merge_two_clusters, get_max_identifier, \
    get_best_distance_cluster_to_cluster, find_closest_cluster
from Framework.Algorithms.CluStream.Cluster import ClustreamCluster as Cluster
from Framework.Algorithms.CluStream.Snapshot import ClustreamSnapshot as Snapshot
from Framework.DatasetReader.DatasetReader import DatasetReader


class Clustream(Algorithm):

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
        self.initialize_clusters()
        self.dataset = self.dataset.iloc[number_of_clusters:]
        self.list_snapshot = []
        self.extra_desc = self.get_string_parameters()

        self.finished = False

    def get_string_parameters(self):
        """
        semplice funzione che si usa generalmente per i titoli dei grafici
        che genera una stringa con i parametri di esecuzione
        """
        return 'Threshold: ' + str(self.threshold) + ', p: ' + str(self.p) + ', N Clusters: ' + str(self.number_of_clusters) + ', window: ' + str(self.time_window_length)

    def get_name(self):
        return 'Clustream'

    def plot_clusters(self, axis, show_snapshots = False):
        """
        funzione, da eseguire ovviamente solo dopo il termine dell'esecuzione dell'algoritmo,
        che plotta le posizioni dei centroidi dei cluster su una variable axis di pyplot ricevuta
        dall'esterno della classe.
        Questa funzione non genera un grafico per conto suo, è semplicemente una funzione helper.
        """
        columns = ['timestamp', 'centroids']
        columns_values = []
        first = True
        for snapshot in tqdm(self.snapshots, desc='Plotting Clustream Clusters'):
            time = snapshot.timestamp
            for cluster in snapshot.clusters:
                columns_values.append([time, float(cluster.centroid_coordinates)])
                if show_snapshots:
                    if first:
                        axis.scatter(time, cluster.centroid_coordinates, c='red', s=15, alpha=1,
                                     label='Cluster Centroid')
                        first = False
                    else:
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
        """
        restituisce una tupla contenente la lista dei centroidi dei microclusters e il timestamp relativo all'inizio
        della finestra temporale nella quale sono stati generati
        """
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
        """
        funzione di inizializzazione, prende i primi n_cluster valori e ne crea un numero uguale di microcluster
        che ne contengono uno ciascuno

        inizializza altre variabili necessarie
        """
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

    def remove_cluster(self, list_cluster: list, list_removed, threshold: int) -> tuple:
        """
        This function implements the remove operation for clustream, the remove
        operation is based on the age.
        """
        oldest_cluster = get_oldest_cluster(list_cluster, threshold)
        if oldest_cluster is not None:
            list_cluster.remove(oldest_cluster)
            list_removed.append(oldest_cluster)
            return True, oldest_cluster
        return False, None

    # Nel caso di clustream non è stato necessario aggiungere una variante con
    # un finestramento differente. Questa funzione esiste solo per rispettare
    # quando richiesto dalla classe astratta Algorithm
    def windowed_run(self):
        self.run()

    def run(self):
        """
        funzione che lancia l'algoritmo
        originariamente era possibile creare un file csv da qua, ma la funzionalità è stata spostata direttamente
        all'interno del file utilities dentro la cartella Old Notebooks
        """
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
                                                                   self.threshold)
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
