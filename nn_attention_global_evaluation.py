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

test_csv_file = "/home/li/torch/csv/test_csv.csv"


if __name__ == '__main__':

    ############## Data Preparation ###################

    model_path = "/home/li/torch/model/attention_net_u_artificial_Q_9_K_6_F_5_REG_L0_ACT_sigmoid_WD_00001_CV.model"
    username = "artificial"

    input_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_X_Y_input.csv"
    output_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_X_Y_output.csv"


    extra = "20190618"
    plot_path = "/home/li/torch/figure/attention_net/output/" + str(extra) + "_object_8_" + str(username) + "_output_mds_figure.png"
    csv_path = "/home/li/torch/figure/attention_net/output/"+ str(extra) + "_object_8_" + str(username) + "_output_distance.csv"

    bar_path = "/home/li/torch/figure/attention_net/distribution/"+ str(extra) + "_bar_graph_" + str(username) + ".png"
    item_name_path = "/home/li/torch/figure/attention_net/distribution/" + str(extra) + "_item_name_" + str(username) + ".txt"


    dataset = GlobalModelDataset(input_csv, output_csv)

    params = (9,6,5)

    model = Attention_Net(dataset, params, activation = ACT)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        
    

    item_list = model.item_list

    print(item_list)
    

    embedding = MDS(n_components = 2, dissimilarity = "precomputed")

    input_sample = random.sample(range(64), OBJECT_NUM)
    #input_sample = [4,14,45,62,35,22,54,23]
        
    input_name_list = []
    for i in input_sample:
        input_name_list.append(item_list[i])

    with open(item_name_path, 'w') as item_f:
        for item in input_name_list:
            item_f.write(str(item) + "\r\n")

    input_test = []
    for item in item_list:
        if item in input_name_list:
            input_test.append(1)
        else:
            input_test.append(0)
    
    input_torch= torch.from_numpy(np.array(input_test)).unsqueeze(0).float()
    output,dist = model.forward(input_torch)
    output_matrix = model.get_output_small_matrix(input_torch, output, pandas = False)
    output_df = model.get_output_small_matrix(input_torch, output, pandas = True)
    pos = embedding.fit_transform(output_matrix)

    dist = list(dist[0].detach().numpy())

    print(dist)

    x_list = []
    y_list = []
    for p in pos:
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list, c = "red", marker = "o")
    for i in range(len(input_name_list)):
        plt.annotate(input_name_list[i], tuple(pos[i]))

    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.savefig(plot_path)
    
    output_df.to_csv(csv_path)

    plt.close('all')

    plt.bar(range(len(dist)), dist, color = 'b')
    plt.savefig(bar_path)
    plt.close('all')

    
    
    

    

        
