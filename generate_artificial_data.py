#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import math
import random
import itertools

#PARAMS
class_number = 4
item_number = 32
total_number = class_number * item_number

class_product_list = [[1,1,1],[1,1,-1],[1,-1,-1],[-1,-1,-1]]
class_list = ["Class 1","Class 2","Class 3","Class 4"]

data_df = pd.DataFrame()
data_info_df = pd.DataFrame()

artificial_data_path = "/home/li/torch/artificial_data/artificial_data.csv"
artificial_data_info = "/home/li/torch/artificial_data/artificial_data_info.csv"


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


################### Generate Dataset
'''
item_class_list = []
item_name_list = []
item_x_list = []
item_y_list = []
item_z_list = []

item_name = 0

for i in range(class_number):
    product_vector = class_product_list[i]
    class_number = class_list[i]
    for j in range(item_number):
        item_name_list.append(item_name)
        item_class_list.append(i)
        random_coord = np.random.rand(3)
        random_coord = np.multiply(random_coord,product_vector)
        item_x_list.append(random_coord[0])
        item_y_list.append(random_coord[1])
        item_z_list.append(random_coord[2])
        item_name += 1

data_info_df["Class"] = class_list
data_info_df["x_low"] = [0,0,0,-1]
data_info_df["x_high"] = [1,1,1,0]
data_info_df["y_low"] = [0,0,-1,-1]
data_info_df["y_high"] = [1,1,0,0]
data_info_df["z_low"] = [0,-1,-1,-1]
data_info_df["z_high"] = [1,0,0,0]
#data_info_df.to_csv(artificial_data_info)

data_df["Item Name"] = item_name_list
data_df["Class"] = item_class_list
data_df["X"] = item_x_list
data_df["Y"] = item_y_list
data_df["Z"] = item_z_list
#data_df.to_csv(artificial_data_path)
'''
#################### Generate input output files

data_number = 10000
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


extra = "_class_1_4_X_Y"

artificial_data_input_csv = "/home/li/torch/artificial_data/artificial_data_" + str(data_number) + str(extra) + "_input.csv"
artificial_data_output_csv = "/home/li/torch/artificial_data/artificial_data_" + str(data_number) + str(extra) + "_output.csv"

data_df = pd.read_csv(artificial_data_path)
data_info_df = pd.read_csv(artificial_data_info)

item_name_list = data_df['Item Name']
x_list = data_df['X']
y_list = data_df['Y']
z_list = data_df['Z']

####### Projection
XOZ_c1_coord_list = x_list * XOY_coord_1_coeff
XOZ_c2_coord_list = z_list * XOY_coord_2_coeff

XOY_c1_coord_list = x_list * XOZ_coord_1_coeff
XOY_c2_coord_list = y_list * XOZ_coord_2_coeff

#print(c1_coord_list)
#print(c2_coord_list)

com = itertools.combinations(range(64),2)
com_list = []
for c in com:
    com_list.append(c)

input_matrix = []
output_matrix = []

for i in range(data_number):
    group_rand = random.randint(0,1)
    if group_rand == 0:
        random_list = random.sample(class_1_list, 8)
        c1_coord_list = XOY_c1_coord_list
        c2_coord_list = XOY_c2_coord_list
    elif group_rand == 1:
        random_list_1 = random.sample(class_1_list,4)
        random_list_2 = random.sample(class_4_list,4)
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

    if i % 100 == 0:
        print(i)
            
data_f_input = pd.DataFrame(input_matrix, columns = class_list, index = range(data_number))
data_f_input.to_csv(artificial_data_input_csv)
data_f_output = pd.DataFrame(output_matrix, columns = com_list, index = range(data_number))
data_f_output.to_csv(artificial_data_output_csv)


