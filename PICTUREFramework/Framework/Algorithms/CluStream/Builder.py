from abc import ABC, abstractmethod

class Builder(ABC):
    @abstractmethod
    def snapshot(self) -> Snapshot:
        pass

    @abstractmethod
    def set_synthesis_list(self, timestamp: int, dimension: list, list_cluster: list,
                           list_cluster_removed: list) -> None:
        pass

    @abstractmethod
    def add_original_data_list(self, list_original_data: dict) -> None:
        pass