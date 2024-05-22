from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def windowed_run(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def plot_clusters(self, axis, show_snaps=False):
        pass

    @abstractmethod
    def get_string_parameters(self):
        pass

    @abstractmethod
    def get_snapshots(self):
        pass
