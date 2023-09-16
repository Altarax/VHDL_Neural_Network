import numpy as np

def find_max_min_list_of_arrays(list_of_arrays):
    min_val = np.inf
    max_val = -np.inf
    for key, arr in list_of_arrays.items():
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_min < min_val:
            min_val = arr_min
        if arr_max > max_val:
            max_val = arr_max
            
    return min_val, max_val