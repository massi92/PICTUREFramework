import sys

import numpy as np

from Framework.Algorithms.CluStream.Cluster import ClustreamCluster as Cluster


def find_closest_cluster(point: list, list_cluster: list) -> tuple:
    """
    This function given a point find the nearest cluster and compute the distance
    :param point:
    :param list_cluster:
    :return:
    """
    min_distance = sys.float_info.max
    closest_cluster = None
    for cluster in list_cluster:
        np_centroid = np.asarray(cluster.centroid)
        np_point = np.asarray(point)
        distance = np.linalg.norm(np_centroid - np_point)
        if distance < min_distance:
            closest_cluster = cluster
            min_distance = distance
    return closest_cluster, min_distance


def get_best_distance_cluster_to_cluster(cluster_selected: Cluster, list_cluster: list):
    min_distance = sys.float_info.max
    for cluster in list_cluster:
        if cluster != cluster_selected:
            distance = np.linalg.norm(cluster.centroid - cluster_selected.centroid)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def get_oldest_cluster(list_cluster: list, threshold: int) -> Cluster:
    """
    This function given a list of cluster get the oldest cluster
    :param list_cluster:
    :param threshold:
    :return:
    """
    min_relevance_timestamp = sys.float_info.max
    oldest_cluster = None
    for cluster in list_cluster:
        relevance_timestamp = cluster.get_relevance_timestamp(m=10)
        if relevance_timestamp < threshold and relevance_timestamp < min_relevance_timestamp:
            min_relevance_timestamp = relevance_timestamp
            oldest_cluster = cluster
    return oldest_cluster


def get_max_identifier(list_cluster: list, list_cluster_rem: list) -> int:
    """
    This function given a list cluster find the max id value and add one to the value
    :param list_cluster:
    :param list_cluster_rem:
    :return:
    """
    list_index = []
    list_index.extend(get_list_id_cluster(list_cluster))
    list_index.extend(get_list_id_cluster(list_cluster_rem))
    return max(list_index) + 1


def get_list_id_cluster(list_cluster: list):
    """
    This function given a list of cluster extract the list of id
    :param list_cluster:
    :return:
    """
    list_id = []
    for cluster in list_cluster:
        list_id.append(cluster.identifier)
    return list_id


def merge_two_clusters(cluster1: Cluster, cluster2: Cluster, list_cluster: list, list_cluster_removed: list) -> Cluster:
    """
    This function implements merge operation given two cluster
    """
    total_nb_points = cluster1.nb_points + cluster2.nb_points
    total_linear_sum = cluster1.linear_sum + cluster2.linear_sum
    total_squared_sum = cluster1.squared_sum + cluster2.squared_sum
    total_linear_time_sum = cluster1.linear_time_sum + cluster2.linear_time_sum
    total_squared_time_sum = cluster1.squared_time_sum + cluster2.squared_time_sum
    id_new_cluster = get_max_identifier(list_cluster, list_cluster_removed)
    new_cluster = Cluster(id_new_cluster, total_nb_points, [], [], total_linear_sum, total_squared_sum,
                          total_linear_time_sum, total_squared_time_sum, cluster1.centroid, 0.0)
    new_cluster.centroid = new_cluster.calculate_centroid()
    new_cluster.radius = new_cluster.calculate_radius()
    if not len(cluster1.id_list_sons) != 0:
        if not len(cluster2.id_list_sons) != 0:
            new_cluster.id_list_sons.extend([cluster1.identifier, cluster2.identifier])
        else:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.append(cluster1.identifier)
            new_cluster.id_list_sons.extend(cluster2.id_list_sons)
    else:
        if len(cluster2.id_list_sons) != 0:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.extend(cluster1.id_list_sons)
            new_cluster.id_list_sons.append(cluster2.identifier)
        else:
            new_cluster.id_list_parents.extend([cluster1.identifier, cluster2.identifier])
            new_cluster.id_list_sons.extend(cluster1.id_list_sons)
            new_cluster.id_list_sons.extend(cluster2.id_list_sons)
    list_cluster.remove(cluster1)
    list_cluster.remove(cluster2)
    list_cluster_removed.extend([cluster1, cluster2])
    list_cluster.append(new_cluster)
    return new_cluster
