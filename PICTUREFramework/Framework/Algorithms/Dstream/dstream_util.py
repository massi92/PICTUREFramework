import random
import numpy as np

def get_sporadic_grids(grids):
    sporadic_grids = {}
    for indices, grid in grids.items():
        if grid.status == 'SPORADIC':
            sporadic_grids[indices] = grid
    return sporadic_grids


def get_last_label_changed(grids):
    last_grids = {}
    for indices, grid in grids.items():
        if grid.label_changed_last_iteration:
            last_grids[indices] = grid
    return last_grids


def get_dense_grids(grids):
    dense_grids = {}
    non_dense_grids = {}
    for indices, grid in grids.items():
        if grid.density_category == 'DENSE':
            dense_grids[indices] = grid
        else:
            non_dense_grids[indices] = grid
    return dense_grids, non_dense_grids


def get_grids_of_cluster_class(grids, class_key):
    found_grids = {}
    for indices, grid in grids.items():
        if grid.label == class_key:
            found_grids[indices] = grid
    return found_grids


def generate_unique_class_key(class_keys):
    test_key = int(np.round(random.uniform(0, 1), 8) * 10 ** 8)
    while test_key in class_keys:
        test_key = np.int(np.round(random.uniform(0, 1), 8) * 10 ** 8)
    return test_key


def reset_last_label_changed(grids):
    for indices, grid in grids.items():
        grid.label_changed_last_iteration = False
        grids[indices] = grid

def get_most_recently_categorically_changed_grids(grids):
    return_grids = {}
    for indices, grid in grids.items():
        if grid.category_changed_last_time == True:
            return_grids[indices] = grid
    return return_grids
