#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import random
import itertools
import math
import os

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.manifold import MDS

from neural_network import Attention_Net
from neural_network import Linear_Net
from datasets import GlobalModelDataset


########### FUNCTIONS

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var,2).sum())

def get_inner_class_distance(df, sample_list, order = 1):
    distance = 0
    list_num = len(sample_list)
    combines = itertools.combinations(sample_list,2)
    com_number = 0
    for combine in combines:
        #print("haha")
        d = df.loc[combine[0],combine[1]]
        com_number += 1
        #print(combine)
        distance += pow(float(d), order)
    #print(com_number)
    distance = distance/float(com_number)
    return distance

def get_inter_class_distance(df, class_1_list, class_2_list, order = 1):
    distance = 0
    class_1_num = len(class_1_list)
    class_2_num = len(class_2_list)
    for name_i in class_1_list:
        inter_d = 0
        for name_j in class_2_list:
            d = df.loc[str(name_i),str(name_j)]
            inter_d += pow(float(d), order)
        inter_d = inter_d/class_2_num
        distance += inter_d
    distance = distance/class_1_num
    return distance



##########################################################################
lifelog_itemlist = "/home/li/datasets/lifelog/itemlist.csv"
lifelog_data = pd.read_csv(lifelog_itemlist)
group_path = "/home/li/datasets/lifelog/Group1_64.txt"
group_list = []
group_item_name_list = []



with open(group_path,"r") as g_f:
    for line in g_f.readlines():
        group_list.append(int(line.strip()))
        group_item_name_list.append(lifelog_data.loc[int(line.strip()) - 1,"Name"])

################## PARAMS

ACT = "sigmoid"
BATCH_SIZE = 1
LEARNING_RATE = 0.02
WEIGHT_DECAY = torch.tensor(0.0001).float()
QUERY_DIM = 5
KEY_DIM = 10
FEATURE_DIM = 8

CV_NUM = 5
OBJECT_NUM = 8

TEST_NUMBER = 100

test_csv_file = "/home/li/torch/csv/test_csv.csv"


if __name__ == '__main__':

    ############## Data Preparation ###################

    model_path = "/home/li/torch/model/Artificial_Data_30000_epoch_100_attention_net_u_artificial_Q_9_K_6_F_5_REG_L2_ACT_sigmoid_WD_000001_CV.model"
    username = "artificial"
    
    input_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_XoY_XoZ_input.csv"
    output_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_XoY_XoZ_output.csv"

    extra = "datanumber_30000_L0/20190701"
    evaluation_path = "/home/li/torch/evaluation/"
    
    plot_path = "/home/li/torch/evaluation/" + str(extra) + "_object_8_" + str(username) + "_output_mds_figure.png"
    csv_path = "/home/li/torch/evaluation/"+ str(extra) + "_object_8_" + str(username) + "_output_distance.csv"

    bar_path = "/home/li/torch/evaluation/"+ str(extra) + "_bar_graph_" + str(username) + ".png"
    item_name_path = "/home/li/torch/evaluation/" + str(extra) + "_item_name_" + str(username) + ".txt"

    coeff_path = "/home/li/torch/artificial_data/coefficient_log_30000_test_L2_WD_00001.txt"

    dataset = GlobalModelDataset(input_csv, output_csv)

    params = (9,6,5)

    model = Attention_Net(dataset, params, activation = ACT)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        
    

    item_list = model.item_list

    class_1_list = item_list[:32]
    class_2_list = item_list[32:]

    print(item_list)
    
    embedding = MDS(n_components = 2, dissimilarity = "precomputed")

    d11_list = []
    d22_list = []
    d11_star_list = []
    d22_star_list = []
    d12_star_list = []
    

    for i in range(TEST_NUMBER):
        ## Group 1

        group1_list = random.sample(class_1_list, 8)

        input_test = []
        for item in item_list:
            if item in group1_list:
                input_test.append(1)
            else:
                input_test.append(0)
    
        input_torch = torch.from_numpy(np.array(input_test)).unsqueeze(0).float()
        output,dist = model.forward(input_torch)
        output_large_matrix = model.get_output_matrix(input_torch, output, pandas = True)
        output_matrix = model.get_output_small_matrix(input_torch, output, pandas = False)
        output_df = model.get_output_small_matrix(input_torch, output, pandas = True)

        d11 = get_inner_class_distance(output_large_matrix, group1_list, order = 1)
        d11_list.append(d11)
        #pos = embedding.fit_transform(output_matrix)
        dist = list(dist[0].detach().numpy())

        csv_path = evaluation_path + "group1_test" + str(i) + ".csv"
        output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group1_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        plt.savefig(bar_path)
        plt.close('all')

        ## Group 2
        group2_list = random.sample(class_2_list, 8)

        input_test = []
        for item in item_list:
            if item in group2_list:
                input_test.append(1)
            else:
                input_test.append(0)

        input_torch = torch.from_numpy(np.array(input_test)).unsqueeze(0).float()
        output,dist = model.forward(input_torch)
        output_large_matrix = model.get_output_matrix(input_torch, output, pandas = True)
        output_matrix = model.get_output_small_matrix(input_torch, output, pandas = False)
        output_df = model.get_output_small_matrix(input_torch, output, pandas = True)

        d22 = get_inner_class_distance(output_large_matrix, group2_list, order = 1)
        d22_list.append(d22)
        #pos = embedding.fit_transform(output_matrix)
        dist = list(dist[0].detach().numpy())

        csv_path = evaluation_path + "group2_test" + str(i) + ".csv"
        output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group2_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        plt.savefig(bar_path)
        plt.close('all')


        ## Group 3

        group31_list = random.sample(group1_list, 4)
        group32_list = random.sample(group2_list, 4)
        group3_list = group31_list + group32_list

        input_test = []
        for item in item_list:
            if item in group3_list:
                input_test.append(1)
            else:
                input_test.append(0)

        input_torch = torch.from_numpy(np.array(input_test)).unsqueeze(0).float()
        output,dist = model.forward(input_torch)
        output_large_matrix = model.get_output_matrix(input_torch, output, pandas = True)
        output_matrix = model.get_output_small_matrix(input_torch, output, pandas = False)
        output_df = model.get_output_small_matrix(input_torch, output, pandas = True)

        
        d11_star = get_inner_class_distance(output_large_matrix, group31_list, order = 1)
        d11_star_list.append(d11_star)
        d22_star = get_inner_class_distance(output_large_matrix, group32_list, order = 1)
        d22_star_list.append(d22_star)
        d12_star = get_inter_class_distance(output_large_matrix, group31_list, group32_list, order = 1)
        d12_star_list.append(d12_star)

        dist = list(dist[0].detach().numpy())

        csv_path = evaluation_path + "group3_test" + str(i) + ".csv"
        output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group3_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        plt.savefig(bar_path)
        plt.close('all')

    d11_mean = np.mean(d11_list)
    d22_mean = np.mean(d22_list)
    d11_star_mean = np.mean(d11_star_list)
    d22_star_mean = np.mean(d22_star_list)
    d12_star_mean = np.mean(d12_star_list)

    c1 = d11_star_mean / d11_mean
    c2 = d22_star_mean / d22_mean
    c3 = (d11_star_mean + d22_star_mean)/ (2 * d12_star_mean)

    info0 = "Model: " + str(model_path)
    info01 = "d11 : " + str(d11_mean) + " , d22 : " + str(d22_mean)
    info02 = "d11* : " + str(d11_star_mean) + " , d22* : " + str(d22_star_mean)
    info03 = "d12* : " + str(d12_star_mean)
    info1 = "c1: " + str(c1)
    info2 = "c2: " + str(c2)
    info3 = "c3: " + str(c3)

    with open(coeff_path, "w") as log_f:
        log_f.write(info0 + "\r\n")
        log_f.write(info01 + "\r\n")
        log_f.write(info02 + "\r\n")
        log_f.write(info03 + "\r\n")
        log_f.write(info1 + "\r\n")
        log_f.write(info2 + "\r\n")
        log_f.write(info3 + "\r\n")
    

        
