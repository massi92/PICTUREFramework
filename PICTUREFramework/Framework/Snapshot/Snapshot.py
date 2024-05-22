from abc import ABC, abstractmethod


class Snapshot(ABC):
    @abstractmethod
    def list_cluster(self):
        pass

    @property
    @abstractmethod
    def clusters(self):
        pass