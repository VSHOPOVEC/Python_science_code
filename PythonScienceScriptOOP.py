import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math as mt
from scipy.special import erf
static_const_Norm = 10

name_of_files = [
                 "F27_data_arr.dat", "F30_data_arr.dat", "F32_data_arr.dat", "F35_data_arr.dat",
                 "F37_data_arr.dat", "F40_data_arr.dat",
                 "F42_data_arr.dat", "F45_data_arr.dat", "F47_data_arr.dat", "F50_data_arr.dat"]

def load_file_data(file_path):
    try:
        with open(file_path) as file:
            input_file = [file_path, [], []]
            array_of_strings = file.readlines()
            for line in array_of_strings:
                split_line = line.split()
                if split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN":
                    input_file[1].append(float(split_line[0]))  # time
                    input_file[2].append(float(split_line[2]))  # point
        return input_file
    except ValueError:
        print("Can't open your file")


def load_file_data_with_map(file_path):
    with open(file_path) as file:
        mydict = {}
        array_of_strings = file.readlines()
        for line in array_of_strings:
            split_line = line.split()
            if split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN" and split_line[
                0] not in mydict:
                results = []
                mydict[split_line[0]] = results
                results.append(float(split_line[2]))
            elif split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN":
                mydict[split_line[0]].append(float(split_line[2]))
    time_list = list(mydict.keys())
    value_list = list(mydict.values())
    for index in range(len(time_list)):
        time_list[index] = float(time_list[index])
        value_list[index] = np.array(list(value_list[index]))
    time_list = np.array(time_list)
    sorted_list = list(zip(time_list, value_list))
    sorted_combined = sorted(sorted_list)
    sorted_time, sorted_ch_wait = zip(*sorted_combined)
    return np.array(sorted_time), sorted_ch_wait


def to_sort_points(list_from_file, amount_of_array):
    len_time_arr = len(list_from_file) - 1
    list_result = list(
        np.concatenate([list_from_file[index1] for index1 in range(index2, amount_of_array + index2)]).tolist() for
        index2 in range(0, len_time_arr - amount_of_array))
    return list_result

def to_sort_time(list_time_from_file, amount_of_array):
    len_time_arr = len(list_time_from_file) - 1
    temp_list_time = [np.mean(list_time_from_file[0 + index: amount_of_array + index]) for index in
                      range(0, len_time_arr - amount_of_array)]
    time_result = np.array([float(x) for x in temp_list_time])
    return time_result

def confidence_interval(list_sorted_from_file):
    std = np.array([np.std(item) for item in list_sorted_from_file])
    len_lists = np.array([len(item) for item in list_sorted_from_file])
    sqrt_len_lists = np.sqrt(num)
    coefficient = np.array([get_value(item) for item in len_lists])
    temp_result = list(coefficient * std / sqrt_len_lists)
    result = np.array([float(item) for item in temp_result])
    return result

def to_normalize_data(sub_list_of_sorted_file):
    try:
       max_value = float(np.max(sub_list_of_sorted_file))
       min_value = float(np.min(sub_list_of_sorted_file))
       normal_list = np.array([(item - min_value) / (max_value - min_value) * static_const_Norm for item in sub_list_of_sorted_file])
       return normal_list, (min_value, max_value)
    except ValueError:
        return _,(_,_)

def erf_func(x, sigma, nu):
    eps = 0.00000000001
    return 1 / 2 + 1 / 2 * erf(np.log(x + eps) - nu) / (sigma * np.sqrt(2))

    # Epsilon is a constant need to us,
    # just because in another case u will get error log(0),
    # because min value in array transform into 0.

def liner_func(x, a, b):
    return a * x + b

def exp_func(x, a, b):
    d = 1000 * x
    return a*np.exp(d) + b

def poly_func(x,a,b):
    return a*x**3 + b*x**2

def poly_func_fit(data_x, data_y):
    params = curve_fit(poly_func,data_x, data_y)
    return params

def exp_func_fit(data_x, data_y):
    params = curve_fit(exp_func, data_x, data_y)
    return params


def liner_func_fit(data_x, data_y):
    params = curve_fit(liner_func, data_x, data_y)
    return params

def erf_func_fit(data_x, data_y, min_value, max_value):
    params = curve_fit(erf_func, data_x, data_y)
    sigma_fit, nu_fit = params[0]
    sigma, nu = float(sigma_fit), float(nu_fit)
    check_wait = (mt.exp(nu + (sigma ** 2) / 2) / static_const_Norm) * (max_value - min_value) + min_value
    dispersion = ((mt.exp(sigma ** 2) - 1) * mt.exp(2 * nu + sigma ** 2)) / static_const_Norm * (
        max_value - min_value) + min_value
    return check_wait, dispersion, sigma, nu

def get_value(r_input):
    data = {
        5: 2.78, 6: 2.57, 7: 2.45, 8: 2.37, 9: 2.31, 10: 2.26,
        11: 2.23, 12: 2.20, 13: 2.18, 14: 2.16, 15: 2.15,
        16: 2.13, 17: 2.12, 18: 2.11, 19: 2.10, 20: 2.093,
        25: 2.064, 30: 2.045, 35: 2.032, 40: 2.023, 45: 2.016,
        50: 2.009, 60: 2.001, 70: 1.996, 80: 1.991, 90: 1.987,
        100: 1.984, 120: 1.980
    }

    if r_input <= 20:
        return data.get(r_input, 2.78)

    keys = sorted(data.keys())
    closest_key = min(keys, key=lambda x: abs(x - r_input))
    return data[closest_key]

def to_get_funk_of_prob(normalize_list, max_min_list):
    bins = 100
    min_value, max_value = max_min_list
    edges = np.linspace(0, static_const_Norm, num=bins)
    num_points_into_interval = np.histogram(normalize_list, bins=edges)[0]
    total_sum_of_points = np.sum(num_points_into_interval)
    probability_to_be_in_interval = num_points_into_interval / total_sum_of_points
    y_funk_of_probability = np.cumsum(probability_to_be_in_interval)
    x_funk_of_probability = edges[:-1]
    check_wait, dispersion, sigma, nu = erf_func_fit(x_funk_of_probability, y_funk_of_probability, min_value,
                                                     max_value)
    return y_funk_of_probability, x_funk_of_probability, check_wait, dispersion, sigma, nu


def processing_the_result(file_path, limit_of_thikness, limit_of_time, amount_of_array):
    name, times_c, points_c = load_file_data(file_path)

    time, points = load_file_data_with_map(file_path)
    points = to_sort_points(points, amount_of_array)

    point_mask = [np.array([limit_of_thikness > temp_item for temp_item in item]) for item in points]
    points = [np.array(arr)[mask_temp] for arr, mask_temp in zip(points, point_mask)]
    times = to_sort_time(time, amount_of_array)
    time_mask = np.array([limit_of_time > time for time in times])
    times = times[time_mask]
    try:
       normalize_points = [to_normalize_data(item)[0] for item in points]  # Value of points
       min_max_normalize_points = [to_normalize_data(item)[1] for item in points]  # Value of max and min in each array
       funk_of_res_for_all_arrays = [to_get_funk_of_prob(normal, val_normal) for normal, val_normal in zip(normalize_points, min_max_normalize_points)]

       check_wait = [item[2] for item in funk_of_res_for_all_arrays]
       check_wait = np.array(check_wait)[time_mask]
       dispersion = [item[3] for item in funk_of_res_for_all_arrays]
       dispersion = np.array(dispersion)[time_mask]
       return times, dispersion, check_wait, points, times_c, points_c, name
    except NameError:
        return [],[],[],[],[],[], name

mlist = []
for name in name_of_files:
   _,_,ch,*_  = processing_the_result(name_of_files,10,10,3)
   mlist.append(ch[0])
x = [ ind for ind in range(1, len(name_of_files) + 1)]
plt.plot(x, mlist)
