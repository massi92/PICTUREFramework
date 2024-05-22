from Framework.Snapshot.Snapshot import Snapshot
from copy import copy


class DenStreamSnapshot(Snapshot):
    def __init__(self, p_micro_clusters, o_micro_clusters, timestamp):
        self.clusters = []
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.timestamp = timestamp
        for cluster in p_micro_clusters:
            self.clusters.append(copy(cluster))
            self.p_micro_clusters.append(copy(cluster))
        for cluster in o_micro_clusters:
            self.clusters.append(copy(cluster))
            self.o_micro_clusters.append(copy(cluster))

    @property
    def list_cluster(self):
        return self._clusters

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, value):
        self._clusters = value
