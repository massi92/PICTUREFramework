import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from math import ceil
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pandas as pd

from Framework.Algorithms.Algorithm import Algorithm
from Framework.Algorithms.DenStream.Cluster import MicroCluster
from Framework.DatasetReader.DatasetReader import DatasetReader
from Framework.Algorithms.DenStream.Snapshot import DenStreamSnapshot as Snapshot


def get_nearest_micro_cluster(sample, micro_clusters):
    smallest_distance = sys.float_info.max
    nearest_micro_cluster = None
    nearest_micro_cluster_index = -1
    for i, micro_cluster in enumerate(micro_clusters):
        current_distance = np.linalg.norm(micro_cluster.center() - sample)
        if current_distance < smallest_distance:
            smallest_distance = current_distance
            nearest_micro_cluster = micro_cluster
            nearest_micro_cluster_index = i
    return nearest_micro_cluster_index, nearest_micro_cluster


class DenStream(Algorithm):

    def __init__(self, dataset_reader: DatasetReader, lambd=1.8, eps=5, beta=1, mu=1.5, window=0):
        """
        :param lambd:
        :param eps:
        :param beta:
        :param mu:
        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.
        """
        self.dataset_reader = dataset_reader
        self.lambd = lambd  # rinominare in lambda non è possibile essendo una keyword di python...
        self.eps = eps  # number of core points of DBSCAN
        self.beta = beta
        self.mu = mu
        self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.window = window
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

        self.snapshots = []
        self.i = 0

    def get_string_parameters(self):
        return 'Lambda: ' + str(self.lambd) + ', Epsilon: ' + str(self.eps) + ', Beta: ' + str(
            self.beta) + ', Mu: ' + str(self.mu) + ', Window: ' + str(self.window)

    def get_name(self):
        return 'DenStream'

    def plot_clusters(self, axis, show_snaps=False):
        columns = ['timestamp', 'centroid', 'radius']
        columns_values = []
        for snapshot in tqdm(self.snapshots, desc='Plotting DenStream Clusters'):
            time = snapshot.timestamp
            for cluster in snapshot.clusters:
                columns_values.append([time, float(cluster.centroid_coordinates), float(cluster.weight())])
                if show_snaps:
                    axis.scatter(time, cluster.centroid_coordinates, c='red', alpha=1, s=15)
        return pd.DataFrame(data=columns_values, columns=columns)

    def reset_dataset(self):
        self.dataset_reader.reset_index()

    def get_snapshots(self):
        return self.snapshots

    def reset_parameters(self):
        self.p_micro_clusters = []
        self.o_micro_clusters = []

    def windowed_run(self):
        while self.dataset_reader.has_n_more_samples(self.window) is True:
            self.partial_fit(self.dataset_reader.get_n_next_samples(self.window), None, None, True)

    def run(self):
        self.partial_fit()

    def partial_fit(self, X=None, y=None, sample_weight=None, windowed=False):
        """
        Online learning.      -> which from the paper is the part where
                              -> the microclusters are created and maintained
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        if X is None:
            X = check_array(self.dataset_reader.get_dataset(), dtype=np.float64, order="C")
        else:
            X = check_array(X, dtype=np.float64, order="C")
        n_samples, _ = X.shape
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)
        if not windowed:
            self.i = 0
        else:
            self.reset_parameters()

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
            # if self.i % self.window == 0 and self.i != 0:
            self.i += 1
            if (self.i % self.window == 0 and self.i != 0) or (self.i + 1 % self.window == 0 and windowed):
                self.snapshots.append(Snapshot(self.p_micro_clusters, self.o_micro_clusters, self.i))
        return self

    # più di tanto questa funzione non ci interessa visto che fa la seconda parte
    # di macroclustering e successiva catalogazione
    def fit_predict(self, y=None, sample_weight=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        X = check_array(self.dataset_reader.get_dataset(), dtype=np.float64, order="C")
        '''
        the input is checked to be a non-empty 2D array containing
        only finite values. If the dtype of the array is object, attempt
        converting to float, raising on failure  <- from sklearn
        '''
        n_samples, _ = X.shape
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)
        i = 0
        for sample, weight in zip(X, sample_weight):  # zip associa in ordine un elemento di X e uno di sample weight
            self._partial_fit(sample, weight)
            if i % 100 == 0:
                self.snapshots.append(Snapshot(self.p_micro_clusters, i))
            i += 1
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]

        dbscan = DBSCAN(eps=5, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)
        y = []
        for sample in X:
            index, _ = get_nearest_micro_cluster(sample, self.p_micro_clusters)
            y.append(dbscan.labels_[index])
        return y

    # questa funzione si inserisce dentro la clusse microcluster..
    def _try_merge(self, sample, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, weight)
                return True
        return False

    def _merging(self, sample, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, weight)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, weight):
        self._merging(sample, weight)
        if self.t % self.tp == 0:
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]  # here it removes unwanted p_micro_clusters
            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight
