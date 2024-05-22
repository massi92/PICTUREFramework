import math

from Framework.Cluster.Cluster import Cluster
import numpy as np


class ClustreamCluster(Cluster):
    def __init__(self,
                 identifier=None,
                 nb_points=None,
                 id_list_parents=None,
                 id_list_sons=None,
                 linear_sum=None,
                 squared_sum=None,
                 linear_time_sum=None,
                 squared_time_sum=None,
                 centroid=None,
                 radius=None):
        self.identifier = identifier
        self.nb_points = nb_points
        self.id_list_parents = id_list_parents
        self.id_list_sons = id_list_sons
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.linear_time_sum = linear_time_sum
        self.squared_time_sum = squared_time_sum
        self.centroid = centroid
        self.radius = radius

    def insert_point_cluster(self, new_point: list, current_timestamp: float) -> None:
        self.nb_points += 1
        self.linear_sum = np.add(self.linear_sum, new_point)
        self.squared_sum = self.squared_sum + np.sum(np.square(new_point))
        self.linear_time_sum += current_timestamp
        self.squared_time_sum += math.pow(current_timestamp, 2)
        self.centroid = self.calculate_centroid()
        self.radius = self.calculate_radius()
        self.last_update_timestamp = current_timestamp

    def calculate_centroid(self) -> list:
        centroid = self.linear_sum / self.nb_points
        return centroid

    @property
    def centroid(self):
        return self.calculate_centroid()

    @centroid.setter
    def centroid(self, value):
        self._centroid = value

    @property
    def centroid_coordinates(self):
        return self.calculate_centroid()

    def calculate_radius(self) -> float:
        squared_linear_sum = np.sum(np.square(self.linear_sum))
        result = self.squared_sum / self.nb_points - squared_linear_sum / math.pow(self.nb_points, 2)
        radius = np.sqrt(abs(result))
        return radius

    def get_mu_time(self) -> float:
        return self.linear_time_sum / self.nb_points

    def get_sigma_time(self) -> float:
        return math.sqrt(self.squared_time_sum / self.nb_points - math.pow((self.linear_time_sum / self.nb_points), 2))

    @staticmethod
    def inverse_error(x):
        z = (math.sqrt(math.pi) * x)
        inv_error = z / 2
        z_prod = math.pow(z, 3)
        inv_error += (1 / 24) * z_prod
        z_prod *= math.pow(z, 2)
        inv_error += (7 / 960) * z_prod
        z_prod = math.pow(z, 2)
        inv_error += (127 * z_prod) / 80640
        z_prod = math.pow(z, 2)
        inv_error += (4369 / z_prod) * 11612160
        z_prod = math.pow(z, 2)
        inv_error += (34807 / z_prod) * 364953600
        z_prod = math.pow(z, 2)
        inv_error += (20036983 / z_prod) * 797058662400
        return z_prod

    def get_quantile(self, x):
        assert (0 <= x <= 1)
        return math.sqrt(2) * self.inverse_error(2 * x - 1)

    def get_relevance_timestamp(self, m: int) -> float:
        if self.nb_points < 2 * m:
            return self.get_mu_time()
        return self.get_mu_time() + self.get_sigma_time() * self.get_quantile(m / (2 * self.nb_points))

    def __str__(self):
        return f"\nId cluster: {self.identifier} \nNumber of points inside cluster: " \
               f"{self.nb_points} \nId aggregate clusters parents: {self.id_list_parents} " \
               f"\nId aggregate clusters sons: {self.id_list_sons} " \
               f"\nValue linear vector sum: {self.linear_sum} \nValue squared vector sum: {self.squared_sum} " \
               f"\nTime linear sum: {self.squared_sum} \nTime squared sum: {self.squared_time_sum}" \
               f"\nCentroid: {self.centroid} \nRadius: {self.radius}"

    def to_csv(self):
        return [self.identifier, self.nb_points, self.id_list_parents, self.id_list_sons,
                self.linear_sum, self.squared_sum, self.squared_sum, self.squared_time_sum,
                self.centroid, self.radius]
