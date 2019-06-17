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

data_file_path = "/home/li/torch/data/Group1_nakamura_no_164_20190605.csv"

train_data = pd.read_csv(data_file_path)

with open(group_path,"r") as g_f:
    for line in g_f.readlines():
        group_list.append(int(line.strip()))
        group_item_name_list.append(lifelog_data.loc[int(line.strip()) - 1,"Name"])

data_f = pd.DataFrame({"Index":range(len(group_list)),
                       "ItemID":group_list,
                       "ItemName":group_item_name_list})

#print(data_f)
#data_f.to_csv("/home/li/torch/data/Group1_Map.csv")

item_number = len(group_list)
item_each = 8
data_number = int(len(train_data)/item_each)
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
        point_list.append([float(row["X-Coordinate"]), float(row["Y-Coordinate"])])

    #labels = hierarchical_clustering(point_list)
    
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
            c1 = (Xcoordinate_list[data_index_i],Ycoordinate_list[data_index_i])
            c2 = (Xcoordinate_list[data_index_j],Ycoordinate_list[data_index_j])
            output_array.append(float(get_distance(c1,c2)))
        else:
            output_array.append(np.nan)

    output_matrix.append(output_array)



print(np.array(input_matrix).shape)
print(np.array(output_matrix).shape)

input_csv = "/home/li/torch/data/Data_Input_200_li_mofei_20190531.csv"
output_csv = "/home/li/torch/data/Data_Output_200_li_mofei_20190531.csv"


data_f_input = pd.DataFrame(input_matrix, columns = group_item_name_list, index = range(data_number))
data_f_input.to_csv("/home/li/torch/data/Data_Input_164_nakamura_20190605.csv")
data_f_output = pd.DataFrame(output_matrix, columns = com_list, index = range(data_number))
data_f_output.to_csv("/home/li/torch/data/Data_Output_164_nakamura_20190605.csv")


#in_df = pd.read_csv(input_csv)
#ou_df = pd.read_csv(output_csv)



