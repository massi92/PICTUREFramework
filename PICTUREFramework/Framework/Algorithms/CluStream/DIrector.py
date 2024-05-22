from Framework.Algorithms.CluStream.Builder import Builder


class Director:
    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        self._builder = builder

    def build_minimal_snapshot(self, timestamp: int, dimension: list, list_cluster: list,
                               list_cluster_removed: list) -> None:
        """
        This function create a minimal snapshot, a minimal snapshot has no reference to original data
        """
        self._builder.set_synthesis_list(timestamp, dimension, list_cluster, list_cluster_removed)

    def build_full_snapshot(self, timestamp: int, dimension: list, list_cluster: list, list_cluster_removed: list,
                            list_original_data: dict) -> None:
        """
        This function create a full snapshot, a full snapshot has reference to original data
        """
        self._builder.set_synthesis_list(timestamp, dimension, list_cluster, list_cluster_removed)
        self._builder.add_original_data_list(list_original_data)