#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import random
import itertools
import math
import os
from sklearn.cluster import AgglomerativeClustering


######### Const
N_CLUSTERS = 2
AFFINITY = "enclidean"
LINKAGE = "average"



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

#################### Hierarchical Clustering
def hierarchical_clustering(point_list):
    clustering = AgglomerativeClustering(affinity = AFFINITY, linkage = LINKAGE, n_clusters = N_CLUSTER)
    clustering.fit(point_list)
    
    return list(cluster.labels_)


lifelog_itemlist = "/home/li/datasets/lifelog/itemlist.csv"
lifelog_data = pd.read_csv(lifelog_itemlist)
group_path = "/home/li/datasets/lifelog/Group1_64.txt"
group_list = []
group_item_name_list = []


DATA_ENLARGE_RATE = 30
NOISE_LEVEL = 0.1
NL = "01"


data_file_path = "/home/li/torch/data/Group1_li_mofei_no_200_20190520.csv"

train_data = pd.read_csv(data_file_path)

with open(group_path,"r") as g_f:
    for line in g_f.readlines():
        group_list.append(int(line.strip()))
        group_item_name_list.append(lifelog_data.loc[int(line.strip()) - 1,"Name"])

print(group_item_name_list)
class_1_name_list = group_item_name_list[:32]
class_2_name_list = group_item_name_list[32:]
print(class_1_name_list)
print(class_2_name_list)


data_f = pd.DataFrame({"Index":range(len(group_list)),
                       "ItemID":group_list,
                       "ItemName":group_item_name_list})

#print(data_f)
#data_f.to_csv("/home/li/torch/data/Group1_Map.csv")

item_number = len(group_list)
item_each = 8
data_number = int(len(train_data)/item_each)
output_data_number = data_number * DATA_ENLARGE_RATE
output_dim = int((item_number * (item_number -1))/2)

input_matrix = []
output_matrix = []
item_index_dic = {}

com = itertools.combinations(range(item_number),2)
com_list = []
for c in com:
    com_list.append(c)


for i in range(item_number):
    item_index_dic[str(group_item_name_list[i])] = int(i)

for i in range(1, data_number + 1):
    data = train_data.loc[train_data["RecordID"] == i]

    item_name_list = []
    Xcoordinate_list = []
    Ycoordinate_list = []

    point_list = []
    
    for index,row in data.iterrows():
        item_name_list.append(str(row["ItemName"]))
        Xcoordinate_list.append(float(row["X-Coordinate"]))
        Ycoordinate_list.append(float(row["Y-Coordinate"]))
        point_list.append((float(row["X-Coordinate"]), float(row["Y-Coordinate"])))        

    
    #labels = hierarchical_clustering(point_list)

    for k in range(DATA_ENLARGE_RATE):

        Xcoordinate_list_random = []
        Ycoordinate_list_random = []

        for j in range(len(Xcoordinate_list)):
            Xcoordinate_list_random.append(Xcoordinate_list[j] + float(NOISE_LEVEL * random.uniform(-1,1)))
            Ycoordinate_list_random.append(Ycoordinate_list[j] + float(NOISE_LEVEL * random.uniform(-1,1)))
                
        input_array = []
        for i in range(item_number):
            if group_item_name_list[i] in item_name_list:
                input_array.append(int(1))
            else:
                input_array.append(int(0))

        input_matrix.append(input_array)

        output_array = []

        for c in com_list:
            i = int(c[0])
            j = int(c[1])
            if input_array[i] == 1 and input_array[j] == 1:
                data_index_i = item_name_list.index(str(group_item_name_list[i]))
                data_index_j = item_name_list.index(str(group_item_name_list[j]))
                c1 = (Xcoordinate_list_random[data_index_i],Ycoordinate_list_random[data_index_i])
                c2 = (Xcoordinate_list_random[data_index_j],Ycoordinate_list_random[data_index_j])
                output_array.append(float(get_distance(c1,c2)))
            else:
               output_array.append(np.nan)

        output_matrix.append(output_array)



print(np.array(input_matrix).shape)
print(np.array(output_matrix).shape)

input_csv = "/home/li/torch/data/Data_Input_R_" + str(DATA_ENLARGE_RATE) + "_NL_" + str(NL) + "_6000_li_mofei_20190708.csv"
output_csv = "/home/li/torch/data/Data_Output_R_" + str(DATA_ENLARGE_RATE) + "_NL_" + str(NL) + "_6000_li_mofei_20190708.csv"

data_f_input = pd.DataFrame(input_matrix, columns = group_item_name_list, index = range(output_data_number))
data_f_input.to_csv(input_csv)
data_f_output = pd.DataFrame(output_matrix, columns = com_list, index = range(output_data_number))
data_f_output.to_csv(output_csv)

#in_df = pd.read_csv(input_csv)
#ou_df = pd.read_csv(output_csv)



