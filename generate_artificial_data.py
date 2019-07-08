#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import math
import random
import itertools
import matplotlib.pyplot as plt


#PARAMS
class_number = 4
item_number = 32
total_number = class_number * item_number

class_product_list = [[0.2,0.2,1],[0.2,1,0.2],[1,0.2,0.2],[0.2,0.2,1]]
class_bias_list = [[0,0,0],[0,0,0],[0,0,0],[0.8,0.8,0]]
class_list = ["Class 1","Class 2","Class 3","Class 4"]

data_df = pd.DataFrame()
data_info_df = pd.DataFrame()

artificial_data_path = "/home/li/torch/artificial_data/artificial_data.csv"
artificial_data_info = "/home/li/torch/artificial_data/artificial_data_info.csv"

artificial_data_plot_path = "/home/li/torch/artificial_data/plot/"

############# FUNCTIONS
def get_distance(c1,c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def from_index_get_ij(index,item_number):
    com = itertools.combinations(range(item_number),2)
    c_list = []
    for c in com:
        c_list.append(c)

    if int(index) < len(c_list):
        return c_list[int(index)]
    else:
        return -1


def from_ij_get_index(i,j,item_number):
    com = itertools.combinations(range(item_number),2)
    c_list = []
    for c in com:
        c_list.append(c)
    ij = (int(i),int(j))
    if ij in c_list:
        return c_list.index(ij)
    else:
        return -1

def get_inner_class_distance(coord_list, order = 1):
    distance = 0
    list_num = len(coord_list)
    combines = itertools.combinations(range(list_num),2)
    com_number = 0
    for combine in combines:
        #print("haha")
        c1 = coord_list[combine[0]]
        c2 = coord_list[combine[1]]
        d = get_distance(c1,c2)
        com_number += 1
        #print(combine)
        distance += pow(float(d), order)
    #print(com_number)
    distance = distance/float(com_number)
    return distance

def get_inter_class_distance(class_1_coord_list, class_2_coord_list, order = 1):
    distance = 0
    class_1_num = len(class_1_coord_list)
    class_2_num = len(class_2_coord_list)
    for i in range(class_1_num):
        inter_d = 0
        for j in range(class_2_num):
            c1 = class_1_coord_list[i]
            c2 = class_2_coord_list[j]
            d = get_distance(c1,c2)
            inter_d += pow(float(d), order)
        inter_d = inter_d/class_2_num
        distance += inter_d
    distance = distance/class_1_num
    return distance


################### Generate Dataset

item_class_list = []
item_name_list = []
item_x_list = []
item_y_list = []
item_z_list = []

item_x_divide_list = []
item_y_divide_list = []
item_z_divide_list = []

item_name = 0

for i in range(class_number):
    product_vector = class_product_list[i]
    bias_vector = class_bias_list[i]
    class_number = class_list[i]

    item_x_class_list = []
    item_y_class_list = []
    item_z_class_list = []
    
    for j in range(item_number):
        item_name_list.append(item_name)
        item_class_list.append(i)
        random_coord = np.random.rand(3)
        random_coord = np.multiply(random_coord,product_vector)
        random_coord = np.add(random_coord,bias_vector)
        item_x_list.append(random_coord[0])
        item_y_list.append(random_coord[1])
        item_z_list.append(random_coord[2])
        item_x_class_list.append(random_coord[0])
        item_y_class_list.append(random_coord[1])
        item_z_class_list.append(random_coord[2])
        item_name += 1
    item_x_divide_list.append(item_x_class_list)
    item_y_divide_list.append(item_y_class_list)
    item_z_divide_list.append(item_z_class_list)

data_info_df["Class"] = class_list
data_info_df["x_low"] = [0,0,0,0.8]
data_info_df["x_high"] = [0.2,0.2,1,1]
data_info_df["y_low"] = [0,0,0,0.8]
data_info_df["y_high"] = [0.2,1,0.2,1]
data_info_df["z_low"] = [0,0,0,0]
data_info_df["z_high"] = [1,0.2,0.2,1]
data_info_df.to_csv(artificial_data_info)

data_df["Item Name"] = item_name_list
data_df["Class"] = item_class_list
data_df["X"] = item_x_list
data_df["Y"] = item_y_list
data_df["Z"] = item_z_list
data_df.to_csv(artificial_data_path)

###PLOT scatter

## XOY
plt_file = artificial_data_plot_path + "XoY_plot.png"
plt.scatter(item_x_divide_list[0], item_y_divide_list[0], label = "Class 1")
plt.scatter(item_x_divide_list[1], item_y_divide_list[1], label = "Class 2")
plt.scatter(item_x_divide_list[2], item_y_divide_list[2], label = "Class 3")
plt.scatter(item_x_divide_list[3], item_y_divide_list[3], label = "Class 4")
plt.title("XoY")
plt.legend(loc = "upper right")
plt.savefig(plt_file)
plt.close()

## XOZ
plt_file = artificial_data_plot_path + "XoZ_plot.png"
plt.scatter(item_x_divide_list[0], item_z_divide_list[0], label = "Class 1")
plt.scatter(item_x_divide_list[1], item_z_divide_list[1], label = "Class 2")
plt.scatter(item_x_divide_list[2], item_z_divide_list[2], label = "Class 3")
plt.scatter(item_x_divide_list[3], item_z_divide_list[3], label = "Class 4")
plt.title("XoZ")
plt.legend(loc = "upper right")
plt.savefig(plt_file)
plt.close()

## YoZ
plt_file = artificial_data_plot_path + "YoZ_plot.png"
plt.scatter(item_y_divide_list[0], item_z_divide_list[0], label = "Class 1")
plt.scatter(item_y_divide_list[1], item_z_divide_list[1], label = "Class 2")
plt.scatter(item_y_divide_list[2], item_z_divide_list[2], label = "Class 3")
plt.scatter(item_y_divide_list[3], item_z_divide_list[3], label = "Class 4")
plt.title("YoZ")
plt.legend(loc = "upper right")
plt.savefig(plt_file)
plt.close()




#################### Generate input output files
data_number = 500
item_number = 64
class_1_list = range(32)
class_2_list = range(32,64)
class_3_list = range(64,96)
class_4_list = range(96,128)

#### Class
class_list = list(class_1_list) + list(class_4_list)

XOY_coord_1_coeff = 1
XOY_coord_2_coeff = 1
XOZ_coord_1_coeff = 1
XOZ_coord_2_coeff = 1
YOZ_coord_1_coeff = 1
YOZ_coord_2_coeff = 1


extra = "_class_1_4_XoY_XoZ"

artificial_data_input_csv = "/home/li/torch/artificial_data/artificial_data_" + str(data_number) + str(extra) + "_input.csv"
artificial_data_output_csv = "/home/li/torch/artificial_data/artificial_data_" + str(data_number) + str(extra) + "_output.csv"

coefficient_log = "/home/li/torch/artificial_data/coefficient_log_" + str(data_number) + ".txt"

data_df = pd.read_csv(artificial_data_path)
data_info_df = pd.read_csv(artificial_data_info)

item_name_list = data_df['Item Name']
x_list = data_df['X']
y_list = data_df['Y']
z_list = data_df['Z']

####### Projection
XOZ_c1_coord_list = x_list * XOZ_coord_1_coeff
XOZ_c2_coord_list = z_list * XOZ_coord_2_coeff

XOY_c1_coord_list = x_list * XOY_coord_1_coeff
XOY_c2_coord_list = y_list * XOY_coord_2_coeff

YOZ_c1_coord_list = y_list * YOZ_coord_1_coeff
YOZ_c2_coord_list = z_list * YOZ_coord_2_coeff



#print(c1_coord_list)
#print(c2_coord_list)

com = itertools.combinations(range(64),2)
com_list = []
for c in com:
    com_list.append(c)

input_matrix = []
output_matrix = []

d11_list = []
d22_list = []
d11_star_list = []
d22_star_list = []
d12_star_list = []

for i in range(data_number):

    ## Group 1
    random_list = random.sample(class_1_list, 8)
    group1_list = random_list
    c1_coord_list = XOY_c1_coord_list
    c2_coord_list = XOY_c2_coord_list

    coord_list = []
    for rand in random_list:
        coord = (float(c1_coord_list[rand]),float(c2_coord_list[rand]))
        coord_list.append(coord)

    input_array = []
    for j in range(len(class_list)):
        if class_list[j] in random_list:
            input_array.append(int(1))
        else:
            input_array.append(int(0))

    input_matrix.append(input_array)

    output_array = []
    
    for c in com_list:
        m = int(c[0])
        n = int(c[1])
        if input_array[m] == 1 and input_array[n] == 1:
            c1 = (float(c1_coord_list[class_list[m]]), float(c2_coord_list[class_list[m]]))
            c2 = (float(c1_coord_list[class_list[n]]), float(c2_coord_list[class_list[n]]))
            output_array.append(float(get_distance(c1,c2)))
        else:
            output_array.append(np.nan)
    output_matrix.append(output_array)

    d11 = get_inner_class_distance(coord_list)
    d11_list.append(d11)

    ## Group 2
    random_list = random.sample(class_4_list, 8)
    group2_list = random_list
    c1_coord_list = YOZ_c1_coord_list
    c2_coord_list = YOZ_c2_coord_list

    coord_list = []
    for rand in random_list:
        coord = (float(c1_coord_list[rand]), float(c2_coord_list[rand]))
        coord_list.append(coord)

    input_array = []
    for j in range(len(class_list)):
        if class_list[j] in random_list:
            input_array.append(int(1))
        else:
            input_array.append(int(0))

    input_matrix.append(input_array)

    output_array = []

    for c in com_list:
        m = int(c[0])
        n = int(c[1])
        if input_array[m] == 1 and input_array[n] == 1:
            c1 = (float(c1_coord_list[class_list[m]]), float(c2_coord_list[class_list[m]]))
            c2 = (float(c1_coord_list[class_list[n]]), float(c2_coord_list[class_list[n]]))
            output_array.append(float(get_distance(c1,c2)))
        else:
            output_array.append(np.nan)
    output_matrix.append(output_array)

    d22 = get_inner_class_distance(coord_list)
    d22_list.append(d22)

    ## Group 3
    random_list_1 = random.sample(group1_list,4)
    random_list_2 = random.sample(group2_list,4)
    random_list = random_list_1 + random_list_2
    c1_coord_list = XOZ_c1_coord_list
    c2_coord_list = XOZ_c2_coord_list
    input_array = []
    for j in range(len(class_list)):
        if class_list[j] in random_list:
            input_array.append(int(1))
        else:
            input_array.append(int(0))

    input_matrix.append(input_array)

    output_array = []

    for c in com_list:
        m = int(c[0])
        n = int(c[1])
        if input_array[m] == 1 and input_array[n] == 1:
            c1 = (float(c1_coord_list[class_list[m]]), float(c2_coord_list[class_list[m]]))
            c2 = (float(c1_coord_list[class_list[n]]), float(c2_coord_list[class_list[n]]))
            output_array.append(float(get_distance(c1,c2)))
        else:
            output_array.append(np.nan)
    output_matrix.append(output_array)

    coord_list_1 = []
    for rand in random_list_1:
        coord = (float(c1_coord_list[rand]), float(c2_coord_list[rand]))
        coord_list_1.append(coord)

    coord_list_2 = []
    for rand in random_list_2:
        coord = (float(c1_coord_list[rand]), float(c2_coord_list[rand]))
        coord_list_2.append(coord)


    d11_star = get_inner_class_distance(coord_list_1)
    d22_star = get_inner_class_distance(coord_list_2)
    d12_star = get_inter_class_distance(coord_list_1, coord_list_2)
    d11_star_list.append(d11_star)
    d22_star_list.append(d22_star)
    d12_star_list.append(d12_star)



d11_mean = np.mean(d11_list)
d22_mean = np.mean(d22_list)
d11_star_mean = np.mean(d11_star_list)
d22_star_mean = np.mean(d22_star_list)
d12_star_mean = np.mean(d12_star_list)

c1 = d11_star_mean / d11_mean
c2 = d22_star_mean / d22_mean
c3 = (d11_star_mean + d22_star_mean)/ (2 * d12_star_mean)

info01 = "d11 : " + str(d11_mean) + " , d22 : " + str(d22_mean)
info02 = "d11* : " + str(d11_star_mean) + " , d22* : " + str(d22_star_mean)
info03 = "d12* : " + str(d12_star_mean)
info1 = "c1: " + str(c1)
info2 = "c2: " + str(c2)
info3 = "c3: " + str(c3)

print(info01)
print(info02)
print(info03)
print(info1)
print(info2)
print(info3)

with open(coefficient_log, "w") as log_f:
    log_f.write(info01 + "\r\n")
    log_f.write(info02 + "\r\n")
    log_f.write(info03 + "\r\n")
    log_f.write(info1 + "\r\n")
    log_f.write(info2 + "\r\n")
    log_f.write(info3 + "\r\n")


data_f_input = pd.DataFrame(input_matrix, columns = class_list, index = range(data_number * 3))
data_f_input.to_csv(artificial_data_input_csv)
data_f_output = pd.DataFrame(output_matrix, columns = com_list, index = range(data_number * 3))
data_f_output.to_csv(artificial_data_output_csv)


