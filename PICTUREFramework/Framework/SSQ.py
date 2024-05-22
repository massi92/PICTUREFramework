import sys

import numpy as np
from numpy import copy
from scipy.stats import stats
from tqdm import tqdm


class Cluster:
    density = None
    centroid_coordinates = None
    status = None
    timestamp = None
    samples = None
    density_category = None

    def __init__(self,
                 density,
                 density_category,
                 centroid_coordinates,
                 status,
                 samples):
        self.density_category = density_category
        self.density = density
        self.centroid_coordinates = centroid_coordinates
        self.status = status
        self.samples = copy(samples)


class Snapshot:  # snapshot
    def __init__(self, grid_matrix, timestamp):
        self.clusters = {}
        self.timestamp = timestamp
        for indices, grid_box in grid_matrix.items():
            self.clusters[indices] = Cluster(grid_box.density, grid_box.density_category, indices, grid_box.status,
                                             grid_box.samples)


def calculate_distance(sample_center, centroid):
    return np.linalg.norm(sample_center - centroid)


def find_closest_cluster(sample, snapshot):
    min_distance = None
    min_distance_cluster = None
    for cluster in snapshot.clusters:
        tmp = calculate_distance(sample, cluster.centroid_coordinates)
        tmp_cluster = cluster
        if min_distance == None:
            min_distance = tmp
            min_distance_cluster = tmp_cluster
            continue
        if min_distance > tmp:
            min_distance = tmp
            min_distance_cluster = tmp_cluster
    return min_distance_cluster, min_distance


# SSQ
def calculate_ssq(dataset, snapshots, skip=0, cumulative=False):
    init_window = 0
    ssq = []
    distance = 0
    for snapshot in tqdm(snapshots, desc='Calculating Traditional SSQ'):
        end_window = snapshot.timestamp
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            closest_cluster, distance1 = find_closest_cluster(dataset.iloc[i], snapshot)
            distance += distance1 ** 2
        if init_window >= skip:
            ssq.append((end_window, distance))
        if cumulative is False:
            distance = 0
        init_window = end_window
    return ssq


# ASSQ
def calculate_ssq_all(dataset, snapshots, skip=0, cumulative=False):
    init_window = 0
    distance = 0
    ssq = []
    for snapshot in tqdm(snapshots, desc='Calculating SSQ of entire Snapshot'):
        end_window = snapshot.timestamp
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance += calculate_distance(dataset.iloc[i], cluster.centroid_coordinates) ** 2
        ssq.append((end_window, distance / (end_window - init_window)))
        if not cumulative:
            distance = 0
        init_window = end_window
    return ssq


def calculate_ssq_corrected(dataset, snapshots, skip=0, cumulative=False):
    init_window = 0
    ssq = []
    distance = 0
    old_cluster_max = None
    old_cluster_min = None
    for snapshot in tqdm(snapshots, desc='Calculating Traditional SSQ with Envelope Correction'):
        end_window = snapshot.timestamp
        cluster_max = float(-sys.maxsize)
        cluster_min = float(sys.maxsize)
        for cluster in snapshot.clusters:
            if cluster.centroid_coordinates > cluster_max:
                cluster_max = cluster.centroid_coordinates
            if cluster.centroid_coordinates < cluster_min:
                cluster_min = cluster.centroid_coordinates

        avg = 0
        counter = 0
        for i in range(init_window, end_window + 1):
            avg += dataset.iloc[i]['number_sold']
            counter += 1
        avg = avg / counter

        if old_cluster_max is not None and old_cluster_min is not None:
            avg_clusters = ((cluster_max + old_cluster_max) + (cluster_min + old_cluster_min)) / 4
        else:
            avg_clusters = avg

        avg_distance = calculate_distance(avg, avg_clusters)
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            closest_cluster, distance1 = find_closest_cluster(dataset.iloc[i], snapshot)
            distance += (distance1 ** 2) * avg_distance
        if init_window >= skip:
            ssq.append((end_window, distance))
        if cumulative is False:
            distance = 0
        init_window = end_window
        old_cluster_min = cluster_min
        old_cluster_max = cluster_max
    return ssq


# da escludere...
def calculate_ssq_widened(dataset, snapshots, skip=0, cumulative=False):
    init_window = 0
    ssq = []
    snap_old = None
    distance = 0
    for snapshot in tqdm(snapshots, desc='Calculating SSQ'):
        end_window = snapshot.timestamp
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            closest_cluster, distance1 = find_closest_cluster(dataset.iloc[i], snapshot)
            if snap_old is not None and i < init_window + (end_window - init_window) / 2:
                tmp1, tmp2 = find_closest_cluster(dataset.iloc[i], snap_old)
            else:
                tmp2 = float(sys.maxsize)
            if distance1 > tmp2:
                distance1 = tmp2
            distance += distance1 ** 2
        if init_window >= skip:
            ssq.append((end_window, distance))
        if cumulative is False:
            distance = 0
        init_window = end_window
        snap_old = snapshot
    return ssq


def calculate_cumulative_ssq_all(dataset, snapshots, skip=0):
    init_window = 0
    distance = 0
    ssq = []
    for snapshot in tqdm(snapshots, desc='Calculating Cumulative SSQ'):
        end_window = snapshot.timestamp
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance += (calculate_distance(dataset.iloc[i], cluster.centroid_coordinates) ** 3)
        ssq.append((end_window, distance / (end_window - init_window)))
    return ssq


# primo tentativo di variante, da lasciare perdere, lascio qua solo nel dubbio che possa tornare utile
def calculate_cumulative_ssq_all_width(dataset, snapshots, skip=0):
    init_window = 0
    distance = 0
    ssq = []
    for snapshot in tqdm(snapshots, desc='Calculating Cumulative SSQ - Alternative Version'):
        end_window = snapshot.timestamp
        max = 0
        min = float(sys.maxsize)
        for cluster in snapshot.clusters:
            if cluster.centroid_coordinates > max:
                max = cluster.centroid_coordinates
            if cluster.centroid_coordinates < min:
                min = cluster.centroid_coordinates

        max_sample = float(-sys.maxsize)
        min_sample = float(sys.maxsize)
        for i in range(init_window, end_window + 1):
            tmp = dataset.iloc[i]['number_sold']
            if tmp < min_sample:
                min_sample = tmp
            if tmp > max_sample:
                max_sample = tmp

        ratio = (max - min) / (max_sample - min_sample)
        ratio2 = np.absolute(1 - ratio)
        ratio = 1 + ratio2
        # if ratio < 1:
        #     ratio = (max_sample - min_sample) / (max - min)

        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance += ((calculate_distance(dataset.iloc[i], cluster.centroid_coordinates)) ** 2) * (ratio)
                # distance += (calculate_distance(dataset.iloc[i], cluster.centroid_coordinates)) * (max - min)
        ssq.append((end_window, distance / (end_window - init_window)))
        # distance = 0
        init_window = end_window
    return ssq


# media di campioni, per i cluster si fa max min con lo snap precedente
def envelope_distance_bias_ssq(dataset, snapshots, cumulative=True, skip=0):
    init_window = 0
    distance = 0
    ssq = []
    old_cluster_max = None
    old_cluster_min = None
    for snapshot in tqdm(snapshots, desc='Calculating Cumulative SSQ - Alternative Version'):
        end_window = snapshot.timestamp
        cluster_max = float(-sys.maxsize)
        cluster_min = float(sys.maxsize)
        for cluster in snapshot.clusters:
            if cluster.centroid_coordinates > cluster_max:
                cluster_max = cluster.centroid_coordinates
            if cluster.centroid_coordinates < cluster_min:
                cluster_min = cluster.centroid_coordinates

        avg = 0
        counter = 0
        for i in range(init_window, end_window + 1):
            avg += dataset.iloc[i]['number_sold']
            counter += 1
        avg = avg / counter

        if old_cluster_max is not None and old_cluster_min is not None:
            avg_clusters = ((cluster_max + old_cluster_max) + (cluster_min + old_cluster_min)) / 4
        else:
            avg_clusters = avg

        avg_distance = calculate_distance(avg, avg_clusters)

        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance += ((calculate_distance(dataset.iloc[i],
                                                 cluster.centroid_coordinates)) ** 2) * avg_distance ** 2
        ssq.append((end_window, distance / (end_window - init_window)))
        if not cumulative:
            distance = 0
        init_window = end_window
        old_cluster_min = cluster_min
        old_cluster_max = cluster_max
    return ssq


# media campioni, per snaphsot si fa la retta tra le media dello snap precedente e del presente
def envelope_distance_bias_ssqv2(dataset, snapshots, cumulative=True, skip=0):
    init_window = 0
    distance = 0
    ssq = []
    old_avg = None
    for snapshot in tqdm(snapshots, desc='Calculating Cumulative SSQ - Alternative Version'):
        end_window = snapshot.timestamp
        cluster_max = float(sys.maxsize)
        cluster_min = float(sys.maxsize)
        for cluster in snapshot.clusters:
            if cluster.centroid_coordinates > cluster_max:
                cluster_max = cluster.centroid_coordinates
            if cluster.centroid_coordinates < cluster_min:
                cluster_min = cluster.centroid_coordinates

        avg = 0
        counter = 0
        for i in range(init_window, end_window + 1):
            avg += dataset.iloc[i]['number_sold']
            counter += 1
        avg = avg / counter

        avg_clusters = 0
        i = 0
        for cluster in snapshot.clusters:
            avg_clusters += cluster.centroid_coordinates
            i += 1
        avg_clusters = avg_clusters / i

        if old_avg is not None:
            avg_clusters = (avg_clusters + old_avg) / 2

        avg_distance = calculate_distance(avg, avg_clusters)

        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance += ((calculate_distance(dataset.iloc[i],
                                                 cluster.centroid_coordinates)) ** 2) * avg_distance ** 1
        ssq.append((end_window, (distance / (end_window - init_window))))
        if not cumulative:
            distance = 0
        init_window = end_window
        old_avg = avg_clusters
    return ssq


# TASSQ
def envelope_distance_bias_ssqv3(dataset, snapshots, cumulative=True, skip=0):
    init_window = 0
    distance = 0
    ssq = []
    old_avg = None
    for snapshot in tqdm(snapshots, desc='Calculating Cumulative SSQ - Alternative Version'):
        end_window = snapshot.timestamp
        cluster_max = -float(sys.maxsize)
        cluster_min = float(sys.maxsize)
        for cluster in snapshot.clusters:
            if cluster.centroid_coordinates > cluster_max:
                cluster_max = cluster.centroid_coordinates
            if cluster.centroid_coordinates < cluster_min:
                cluster_min = cluster.centroid_coordinates

        avg = 0
        counter = 0
        for i in range(init_window, end_window + 1):
            avg += dataset.iloc[i]['number_sold']
            counter += 1
        avg = avg / counter

        avg_clusters = 0
        i = 0
        for cluster in snapshot.clusters:
            avg_clusters += cluster.centroid_coordinates
            i += 1
        avg_clusters = avg_clusters / i

        if old_avg is not None:
            avg_clusters = (avg_clusters + old_avg) / 2

        avg_distance = calculate_distance(avg, avg_clusters)
        distance_temp = 0
        if end_window == 0:
            continue
        for i in range(init_window, end_window + 1):
            if init_window + i <= skip:
                continue
            for cluster in snapshot.clusters:
                distance_temp += ((calculate_distance(dataset.iloc[i],
                                                      cluster.centroid_coordinates)) ** 2)
        if avg_distance < 1:
            avg_distance = 1
        distance += ((distance_temp / len(snapshot.clusters)) * avg_distance)
        ssq.append((end_window, distance))
        if not cumulative:
            distance = 0
        init_window = end_window
        old_avg = avg_clusters
    return ssq


def count_clusters(snaps):
    count = 0
    for snapshot in snaps:
        for _ in snapshot.clusters:
            count += 1
    return count


def plot_ssq(axis, dataset, snaps, color, linewidth, alpha, label, skip=0):
    ssq = calculate_ssq(dataset, snaps, skip=skip)
    X = []
    Y = []
    for s in ssq:
        Y.append(s[1])
        X.append(s[0])
    axis.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha, label=label)


def plot_cumulative_ssq(axis, dataset, snaps, color, linewidth, alpha, label, skip=0):
    ssq = calculate_ssq(dataset, snaps, skip=skip, cumulative=True)
    X = []
    Y = []
    for s in ssq:
        Y.append(s[1])
        X.append(s[0])
    axis.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha, label=label)


def plot_ssq_all(axis, axislog, dataset, snaps, color, linewidth, alpha, label, skip=0):
    ssq = calculate_cumulative_ssq_all_width(dataset, snaps, skip=skip)
    cluster_count = count_clusters(snaps)
    X = []
    Y = []
    for s in ssq:
        Y.append(s[1])
        X.append(s[0])
    axis.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha, label=label + ', n_clusters: ' + str(cluster_count))
    if axislog is not None:
        axislog.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha,
                     label=label + ', n_clusters: ' + str(cluster_count))
        axislog.set_yscale('log')


def plot_ssq_avg_distance(axis, axislog, dataset, snaps, color, linewidth, alpha, label, cumulative=False, skip=0,
                          type='traditional'):
    title = 'Title'
    if type == 'traditional':
        ssq = calculate_ssq(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'Traditional SSQ'
    elif type == 'traditional_correction':
        ssq = calculate_ssq_corrected(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'Traditional SSQ with Envelope Correction'
    elif type == 'average_per_sample':
        ssq = calculate_ssq_all(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'SSQ averaged over all clusters'
    elif type == 'teti1':
        ssq = envelope_distance_bias_ssq(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'SSQ - Corrected'
    elif type == 'teti2':
        ssq = envelope_distance_bias_ssqv2(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'SSQ - Corrected V2'
    elif type == 'teti3':
        ssq = envelope_distance_bias_ssqv3(dataset, snaps, cumulative=cumulative, skip=skip)
        title = 'SSQ - Corrected V3'

    cluster_count = count_clusters(snaps)
    X = []
    Y = []
    for s in ssq:
        Y.append(s[1])
        X.append(s[0])
    axis.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha, label=label + ', n_clusters: ' + str(cluster_count))
    if axislog is not None:
        axislog.plot(X, Y, color=color, linewidth=linewidth, alpha=alpha,
                     label=label + ', n_clusters: ' + str(cluster_count))
        axislog.set_yscale('log')
    return title, ssq


def calculate_total_ssq(dataset, snaps, skip=0, type='teti3'):
    ssq = []
    if type == 'traditional':
        ssq = calculate_ssq(dataset, snaps, cumulative=False, skip=skip)
    elif type == 'traditional_correction':
        ssq = calculate_ssq_corrected(dataset, snaps, cumulative=False, skip=skip)
    elif type == 'average_per_sample':
        ssq = calculate_ssq_all(dataset, snaps, cumulative=False, skip=skip)
    elif type == 'teti1':
        ssq = envelope_distance_bias_ssq(dataset, snaps, cumulative=False, skip=skip)
    elif type == 'teti2':
        ssq = envelope_distance_bias_ssqv2(dataset, snaps, cumulative=False, skip=skip)
    elif type == 'teti3':
        ssq = envelope_distance_bias_ssqv3(dataset, snaps, cumulative=False, skip=skip)
    sum = 0
    for s in ssq:
        sum += s[1]
    return sum
