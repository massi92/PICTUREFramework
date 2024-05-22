from tqdm import tqdm
import pandas as pd
import numpy as np
def get_centroid_list(snapshots):
    matrix_centroid = []
    list_timestamp = []
    for snapshot in tqdm(snapshots, desc="Computing Centroids"):
        list_centroid = []
        list_timestamp.append(snapshot.timestamp)
        for synthesis in snapshot.list_cluster:
            centroid = synthesis.centroid[0]
            list_centroid.append(centroid)
        matrix_centroid.append(list_centroid)
    return matrix_centroid, list_timestamp

def get_pearson_correlation_mse_error(algorithm, dimension: int, str_description: str) -> tuple:


    centroids_density, list_timestamp = get_centroid_list(algorithm.get_snapshots())
    original_data = algorithm.dataset_reader.get_dataset()
    print("Len original_data (get_dataset()): ", len(original_data))	
    original_data = transform_range_function(list_timestamp, original_data, dimension)
    print("Len original_data (after transform_range_function()): ", len(original_data))


    matrix_value = get_min_max_mean_matrix(centroids_density)
    #print("original: ")
    #print(original_data)
    #print("matrix_values: ")
    #print(matrix_value)
    list_corr = []
    list_error = []
    for list_value in tqdm(matrix_value, desc=str_description):
        if np.all(list_value == list_value[0]):
            best_corr = -2
        else:
            print("Len list_value: ", len(list_value))
            print("Len original_data: ", len(original_data))
            corr = np.corrcoef(list_value, original_data)
            best_corr = max(corr[0, 1], corr[1, 0])
        error = get_approximation_error_mse(list_value, original_data)
        list_corr.append(best_corr)
        list_error.append(error)
    return np.asarray(list_corr), np.asarray(list_error)


def transform_range_function(list_timestamp: list, original_function: pd.DataFrame, dimension: int) -> list:

    new_function = []
    print("dimension value: ", dimension)
    print("Len list_timestamp: ", len(list_timestamp))
    print("Shape original_function: ", original_function.shape)
    for timestamp in list_timestamp:
        i = original_function.index.get_loc(timestamp)
        new_function.append(original_function.iloc[i, dimension])
    print("Len new_function: ", len(new_function))
    return new_function


def get_min_max_mean_matrix(list_list_centroid: list) -> list:
    list_min_centroid = []
    list_max_centroid = []
    list_mean_centroid = []
    for list_centroid in list_list_centroid:
        list_min_centroid.append(float(min(list_centroid)))
        list_max_centroid.append(float(max(list_centroid)))
        list_mean_centroid.append(float(np.mean(list_centroid)))
    print("Len list_min_centroid: ", len(list_min_centroid))
    print("Len list_max_centroid: ", len(list_max_centroid))
    print("Len list_mean_centroid: ", len(list_mean_centroid))
    return [list_min_centroid, list_max_centroid, list_mean_centroid]


def get_approximation_error_mse(list_value: list, list_data: list) -> float:
    N = len(list_value)
    squared_sum = 0
    for i, j in zip(list_value, list_data):
        squared_sum = squared_sum + np.sum(np.square(np.asarray(j) - np.asarray(i)))
    mean_squared_error = 1 / N * squared_sum
    return mean_squared_error