import csv

from Framework.Snapshot.Snapshot import Snapshot


class ClustreamSnapshot(Snapshot):
    def __init__(self) -> None:
        self.dimension = None
        self.timestamp = None
        self.list_cluster = None
        self.list_cluster_removed = None
        self.list_original_data = None
        self.list_original_data_removed = None

    def to_csv_list_cluster(self, name: str):
        """
        This function save snapshot information insiede csv file
        :param name:
        """
        header = ["id_cluster", "point_inside", "id_aggregate_cluster_parents", "id_aggregate_cluster_son",
                  "linear_vector_sum", "linear_vector_squared_sum", "linear_time_sum", "linear_square_time_sum"
            , "centroid", "radius"]
        with open(name + ".csv", 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for cluster in self.list_cluster:
                writer.writerow(cluster.to_csv())

    def print_information_snapshot(self):
        """
         This function is for debug
         """
        information = ""
        for cluster in self.list_cluster:
            information = information + str(cluster)
        information = information + "\n-------rimossi--------\n"
        for cluster in self.list_cluster_removed:
            information = information + "\n" + str(cluster.identifier) + ": " + str(cluster.nb_points) + " , " + str(
                cluster.get_relevance_timestamp(10))
        print(information)

    @property
    def clusters(self):
        return self.list_cluster

    def list_cluster(self):
        return self.list_cluster