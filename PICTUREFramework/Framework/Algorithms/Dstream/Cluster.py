from numpy import copy

from Framework.Cluster.Cluster import Cluster


class DStreamCluster(Cluster):
    density = None
    centroid_coordinates = None
    status = None
    timestamp = None
    samples = None
    density_category = None

    def __init__(self,
                 density,
                 density_category,
                 centroid_coordinates,
                 status,
                 samples):
        self.density_category = density_category
        self.density = density
        self.centroid_coordinates = centroid_coordinates
        self.status = status
        self.samples = copy(samples)

    @property
    def centroid(self):
        return [self.centroid_coordinates]