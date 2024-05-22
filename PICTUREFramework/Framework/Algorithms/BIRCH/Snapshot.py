from Framework.Snapshot.Snapshot import Snapshot


class BirchSnapshot(Snapshot):
    def __init__(self, clusters, timestamp):
        self._clusters = clusters
        self.timestamp = timestamp

    @property
    def list_cluster(self):
        return self.clusters

    @property
    def clusters(self):
        return self._clusters