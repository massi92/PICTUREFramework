from Framework.Algorithms.Dstream.Cluster import DStreamCluster
from Framework.Snapshot.Snapshot import Snapshot


class DStreamSnapshot(Snapshot):  # snapshot
    def __init__(self, grid_matrix, timestamp, domains_per_dimension, partition_per_dimension):
        self.clusters = []  # era {}
        self.timestamp = timestamp
        self.dom = domains_per_dimension
        self.part = partition_per_dimension
        self.top = self.dom[0][1]
        self.bottom = self.dom[0][0]
        # for indices, grid_box in grid_matrix.items():
        #     self.clusters[indices] = Cluster(grid_box.density, grid_box.density_category, indices, grid_box.status,
        #                                      grid_box.samples)
        for indices, grid_box in grid_matrix.items():
            self.clusters.append(
                DStreamCluster(grid_box.density, grid_box.density_category, self.convert_indexes(indices),
                               grid_box.status,
                               grid_box.samples))

    def convert_indexes(self, indices):
        return indices[0] / self.part[0] * (self.top - self.bottom) + self.bottom + (self.top - self.bottom) / (
                self.part[0] * 2)

    @property
    def list_cluster(self):
        return self._clusters

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, value):
        self._clusters = value