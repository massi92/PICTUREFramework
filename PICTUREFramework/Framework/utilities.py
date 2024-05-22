from time import time as uncopiablenamefortime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from Framework.SSQ import calculate_ssq, envelope_distance_bias_ssqv3, count_clusters, calculate_ssq_all
from Framework.pearson_mse import get_pearson_correlation_mse_error


variable_to_track_time = 0


def tik():
    global variable_to_track_time
    variable_to_track_time = uncopiablenamefortime()


def tok():
    global variable_to_track_time
    return uncopiablenamefortime() - variable_to_track_time


def get_min_cluster(snapshots):
    mins = []
    x = []
    for snap in snapshots:
        tmp_min = 9999
        for cluster in snap.clusters:
            if tmp_min is None or tmp_min > cluster.centroid_coordinates:
                tmp_min = cluster.centroid_coordinates
        mins.append(float(tmp_min))
        x.append(snap.timestamp)
    return mins, x


def get_max_cluster(snapshots):
    maxs = []
    x = []
    for snap in snapshots:
        tmp_max = -999999
        for cluster in snap.clusters:
            if tmp_max is None or tmp_max < cluster.centroid_coordinates:
                tmp_max = cluster.centroid_coordinates
        maxs.append(float(tmp_max))
        x.append(snap.timestamp)
    return maxs, x


def get_avg_cluster(snapshots):
    mins, _ = get_min_cluster(snapshots)
    maxs, _ = get_max_cluster(snapshots)
    avgs = []
    for min, max in zip(mins, maxs):
        avgs.append((min + max) / 2)
    return avgs, _


def get_actual_avg_cluster(snapshots):
    avgs = []
    x = []
    for snap in snapshots:
        tmp = 0
        i = 0
        for cluster in snap.clusters:
            tmp += cluster.centroid_coordinates
            i += 1
        x.append(snap.timestamp)
        avgs.append(float(tmp / i))
    return avgs, x


def plot_envelope(axis, snapshots):
    mins, x2 = get_min_cluster(snapshots)
    axis.plot(x2, mins, label='Min Cluster', color='darkgreen')
    maxs, x2 = get_max_cluster(snapshots)
    axis.plot(x2, maxs, label='Max Cluster', color='red')
    avgs, x2 = get_avg_cluster(snapshots)
    axis.plot(x2, avgs, label='Cluster Middle Point', color='fuchsia')
    avgs2, x2 = get_actual_avg_cluster(snapshots)
    axis.plot(x2, avgs2, label='Cluster Average', color='springgreen')

    columns = ["timestamp", "mins", 'maxs', 'median', 'average']
    columns_values = []
    for t, x, y, z, j in zip(x2, mins, maxs, avgs, avgs2):
        columns_values.append([t, x, y, z, j])
    return pd.DataFrame(data=columns_values, columns=columns)


from Framework.Algorithms.CluStream.Clustream import Clustream
from Framework.Algorithms.Clustream_density import ClustreamDensity


def launch_algorithm_and_plot_envelope(algorithm, show_snapshots=False, windowed=True, save=False, index=0,
                                       show_dataset=True, show_envelope=True, fig=None, ax1=None):
    if isinstance(algorithm, ClustreamDensity) or isinstance(algorithm, Clustream):
        tik()
        algorithm.run()
    else:
        if windowed:
            tik()
            algorithm.windowed_run()
        else:
            tik()
            algorithm.run()
    algorithm_time = tok()
    data = algorithm.dataset_reader.get_dataset()
    x1 = np.linspace(0, len(data), num=len(data))
    if fig is None and ax1 is None:
        fig, ax1 = plt.subplots(figsize=(22, 9), dpi=400)
    if show_dataset:
        ax1.plot(x1, data[:], alpha=0.5, label='Dataset', color='steelblue')

    if show_envelope:
        df_envelope = plot_envelope(ax1, algorithm.snapshots)
    else:
        df_envelope = None
    df_clusters = None
    df_clusters = algorithm.plot_clusters(ax1, show_snapshots)
    ax1.set_title(
        algorithm.get_name() + ': ' + algorithm.get_string_parameters() + ', Computing Time: ' + "{:.2f}".format(
            algorithm_time) + ' [s]', fontsize=14)
    ax1.legend(fontsize=10, loc=1)
    if save:
        fig.savefig('den1\\' + algorithm.get_name() + '_windowed_' + str(windowed) + str(index).zfill(4) + '.png')
        plt.close()
    return algorithm_time, df_envelope, df_clusters


def full_array_of_tests(algorithm, show_snapshots=False, windowed=True, save_output_images=False, index=0, show_dataset=True,
                        show_envelope=True, fig=None, ax1=None, onlyrun=False, nograph=False, only_tssq=False):
    if isinstance(algorithm, ClustreamDensity) or isinstance(algorithm, Clustream):
        tik()
        algorithm.run()
    else:
        if windowed:
            tik()
            algorithm.windowed_run()
        else:
            tik()
            algorithm.run()
    algorithm_time = tok()

    total_ssq = 0
    total_assq = 0
    total_taassq = 0
    cluster_count = 0
    pearson = None
    if onlyrun is False:
        if not only_tssq:
            ssq = calculate_ssq(algorithm.dataset_reader.get_dataset(), algorithm.snapshots, cumulative=False)
            for s in ssq:
                total_ssq += s[1]

            assq = calculate_ssq_all(algorithm.dataset_reader.get_dataset(), algorithm.snapshots, cumulative=False)
            for s in assq:
                total_assq += s[1]

        taassq = envelope_distance_bias_ssqv3(algorithm.dataset_reader.get_dataset(), algorithm.snapshots,
                                              cumulative=False)

        for s in taassq:
            total_taassq += s[1]
        cluster_count = count_clusters(algorithm.snapshots)

        map = {"x": 0, "y": 1, "z": 2}
        dim = map['x']
        pearson = get_pearson_correlation_mse_error(algorithm, dim, str(algorithm.get_name()))

    data = algorithm.dataset_reader.get_dataset()
    x1 = np.linspace(0, len(data), num=len(data))
    df_envelope = None
    df_clusters = None
    if nograph == False:
        if fig is None and ax1 is None:
            fig, ax1 = plt.subplots(figsize=(22, 9), dpi=300)
        if show_dataset:
            ax1.plot(x1, data[:], alpha=0.5, label='Dataset', color='steelblue')

        if show_envelope:
            df_envelope = plot_envelope(ax1, algorithm.snapshots)
        else:
            df_envelope = None
        df_clusters = None
        df_clusters = algorithm.plot_clusters(ax1, show_snapshots)
        ax1.set_title(
            algorithm.get_name() + ': ' + algorithm.get_string_parameters() + ', Computing Time: ' + "{:.2f}".format(
                algorithm_time) + ' [s]', fontsize=14)
        ax1.legend(fontsize=13, loc=1)
        ax1.set_xlabel("Timestamps", fontsize=15)
        ax1.set_ylabel('Measurement / Value', fontsize=15)
        if save_output_images:
            fig.savefig('den1\\' + algorithm.get_name() + '_windowed_' + str(windowed) + str(index).zfill(4) + '.png')
            plt.close()

    return algorithm_time, df_envelope, df_clusters, total_ssq, total_taassq, cluster_count, pearson, total_assq


def save_to_csv(df_envelope: pd.DataFrame, df_clusters, algorithm, windowed = True):
    name = algorithm.get_name()
    if df_envelope is not None:
        if windowed:
            df_envelope.to_csv('window - accel - ' + name + ' - Envelope.csv')
        else:
            df_envelope.to_csv('accel - ' + name + ' - Envelope.csv')
    if df_clusters is not None:
        if windowed:
            df_clusters.to_csv('window - accel - ' + name + ' - Clusters.csv')
        else:
            df_clusters.to_csv('accel - ' + name + ' - Clusters.csv')