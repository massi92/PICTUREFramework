import numpy as np
import math

from abc import ABC, abstractmethod


class Cluster(ABC):
    @property
    @abstractmethod
    def centroid(self):
        pass
