from Framework.Cluster.Cluster import Cluster


class BirchCluster(Cluster):
    centroid_coordinates = None

    def __init__(self,centroid_coordinates):
        self.centroid_coordinates = centroid_coordinates

    @property
    def centroid(self):
        return [self.centroid_coordinates]