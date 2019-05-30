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


############### Dataset Class #####################
class GlobalModelDataset(Dataset):
    def __init__(self,input_csv,output_csv):
        self.input_data = pd.read_csv(input_csv)
        self.output_data = pd.read_csv(output_csv)

        
        self.item_list = list(self.input_data.columns)[1:]
        self.item_index_dic = {}

        self.input_list = []
        self.output_list = []

        for index,row in self.input_data.iterrows():
            self.input_list.append(list(row)[1:])

        for index,row in self.output_data.iterrows():
            output = list(row)[1:]

            #NaN to 0
            for i in range(len(output)):
                if np.isnan(output[i]):
                    output[i] = float(0)
                else:
                    output[i] = math.log(output[i])
            
            self.output_list.append(output)

            
            
        
        self.item_number = len(self.item_list)
        self.item_each = 8
        self.data_num = len(self.input_list)
        self.output_dim = len(self.output_list[0])

        #print(self.data_num)

        self.com_list = []

        com = itertools.combinations(range(self.item_number),2)
        for c in com:
            self.com_list.append(c)


        for i in range(self.item_number):
            self.item_index_dic[str(self.item_list[i])] = int(i)
                

    def __getitem__(self,index):
        input_item = torch.from_numpy(np.array(self.input_list[index])).float()
        output_item = torch.from_numpy(np.array(self.output_list[index])).float()

        return input_item,output_item

    
    def __len__(self):
        return self.data_num

    def get_distance(self,c1,c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        

    def print_info(self,cmd):
        if cmd == "Input":
            print(self.input_list)
            return self.input_list
        elif cmd == "Output":
            print(self.output_list)
            return self.output_list


    def get_non_zeros(self, array):
        array = list(array)
        non_zero_list = []
        for i in range(len(array)):
            if float(array[i]) != 0:
                non_zero_list.append(int(i))
        return non_zero_list


    def count_non_zeros(self,x):
        no = 0
        for element in x:
            if float(element) == float(0.):
                no += 1
        return len(x) - no

    def get_item_list(self):
        return self.item_list

    def get_parameters(self):
        return (self.item_number , self.item_each , self.data_num , self.output_dim)

    def from_index_get_ij(self,index):
        if int(index) < len(self.com_list) and int(index) >= 0:
            return self.com_list[int(index)]
        else:
            return -1

    def from_ij_get_index(self,i,j):
        if int(i) > int(j):
            ij = (int(j),int(i))
        else:
            ij = (int(i),int(j))
        if ij in self.com_list:
            return self.com_list.index(ij)
        else:


            return -1
            
        

##################### Attention Net Class #######################
class Attention_Net(nn.Module):
    def __init__(self,dataset,params = (5,10,8)):
        super(Attention_Net,self).__init__()
        self.dataset = dataset
        self.item_list = dataset.item_list
        self.item_number = len(self.item_list)
        self.output_dim = int((self.item_number * (self.item_number - 1))/2)

        self.query_dim = int(params[0])
        self.key_dim = int(params[1])
        self.feature_dim = int(params[2])
        self.linear_layer1 = nn.Linear(self.item_number,self.query_dim)
        self.key_matrix = torch.nn.Parameter(torch.randn(self.query_dim,self.key_dim))
        self.value_matrix = torch.nn.Parameter(torch.randn(self.key_dim,self.feature_dim))
        self.linear_layer2 = nn.Linear(self.feature_dim, self.output_dim)

        #Initialization
        init.xavier_uniform(self.linear_layer1.weight)
        init.xavier_uniform(self.linear_layer2.weight)
        init.normal(self.linear_layer1.bias,mean = 0,std = 1)
        init.normal(self.linear_layer2.bias,mean = 0,std = 1)
        
        
        

    def forward(self,x):
        #Encoder
        mask = self.get_mask(x)
        #print(mask.shape)
        
        x = self.linear_layer1(x)
        x = x.mm(self.key_matrix)
        x = F.softmax(x,dim = 1)
        x = x.mm(self.value_matrix)

        #Decoder
        x = self.linear_layer2(x)
        x = x.mul(mask)

        return x
        


    def get_mask(self,x):
        #print(x.shape)
        
        mask = []
        x = x.data.numpy()

        
        for batch in x:
            sub_mask = []
            for com in self.dataset.com_list:
                i = com[0]
                j = com[1]
                if int(batch[i]) == 1 and int(batch[j]) == 1:
                    sub_mask.append(1)
                else:
                    sub_mask.append(0)

            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        #print(mask.shape)
        return mask

    def get_output_mask(self,x):

        mask = []
        x = x.data.numpy()

        
        for batch in x:
            sub_mask = []
            for com in self.dataset.com_list:
                i = com[0]
                j = com[1]
                if int(batch[i]) == 1 and int(batch[j]) == 1:
                    sub_mask.append(1)
                else:
                    sub_mask.append(np.nan)

            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        #print(mask.shape)
        return mask

    def get_output_matrix(self, inp, output, pandas = False):

        output_mask = self.get_output_mask(inp)
        output = output.mul(output_mask)
        
        output = list(output[0].detach().numpy())
        #print(len(output))
        output_matrix = np.zeros([self.item_number,self.item_number], dtype = float)
        for com in self.dataset.com_list:
            i = com[0]
            j = com[1]
            output_matrix[i,j] = math.exp(output[dataset.from_ij_get_index(i,j)])
            output_matrix[j,i] = output_matrix[i,j]

        if pandas == False:
            return torch.from_numpy(output_matrix)
        else:
            return pd.DataFrame(output_matrix, columns = self.item_list, index = self.item_list)




########### FUNCTIONS

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var,2).sum())

def get_inner_class_distance(df, sample_list, order = 2):
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

def get_inter_class_distance(df, class_1_list, class_2_list, order = 2):
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

def evaluate_model_inner_inter_distance(model, sample_number = 10, combines = (4,4), order = 2):
    class_1_inner_distance_list = []
    class_2_inner_distance_list = []
    class_1_star_inner_distance_list = []
    class_2_star_inner_distance_list = []
    inter_distance_list = []
    class_1_num = 8
    class_2_num = 8
    class_1_star_num = combines[0]
    class_2_star_num = combines[1]
    name_list = model.item_list
    
    for i in range(sample_number):
        class_1_sample = random.sample(range(32),class_1_num)
        class_2_sample = random.sample(range(32,64),class_2_num)
        class_1_star_sample = random.sample(class_1_sample, class_1_star_num)
        class_2_star_sample = random.sample(class_2_sample, class_2_star_num)
        star_sample = class_1_star_sample + class_2_star_sample

        cross_sample_name_list = []
        class_1_sample_name_list = []
        class_2_sample_name_list = []
        class_1_star_sample_name_list = []
        class_2_star_sample_name_list = []

        for i in class_1_star_sample:
            class_1_star_sample_name_list.append(name_list[i])
        for i in class_2_star_sample:
            class_2_star_sample_name_list.append(name_list[i])
        for i in star_sample:
            cross_sample_name_list.append(name_list[i])
        for i in class_1_sample:
            class_1_sample_name_list.append(name_list[i])
        for i in class_2_sample:
            class_2_sample_name_list.append(name_list[i])


        ## Test for debug
        #print(class_1_star_sample_name_list)
        #print(class_2_star_sample_name_list)
        #print(cross_sample_name_list)
        #print(class_1_sample_name_list)
        #print(class_2_sample_name_list)


        
        cross_test_input = []
        for name in name_list:
            if name in cross_sample_name_list:
                cross_test_input.append(1)
            else:
                cross_test_input.append(0)


        class_1_test_input = []
        for name in name_list:
            if name in class_1_sample_name_list:
                class_1_test_input.append(1)
            else:
                class_1_test_input.append(0)

        class_2_test_input = []
        for name in name_list:
            if name in class_2_sample_name_list:
                class_2_test_input.append(1)
            else:
                class_2_test_input.append(0)

        ### Debug
        #print(len(cross_test_input))
        #print(sum(cross_test_input))
        #print(len(class_1_test_input))
        #print(sum(class_1_test_input))
        #print(len(class_2_test_input))
        #print(sum(class_2_test_input))

        ##### Create Input
        cross_test_input = torch.from_numpy(np.array(cross_test_input)).unsqueeze(0).float()
        class_1_test_input = torch.from_numpy(np.array(class_1_test_input)).unsqueeze(0).float()
        class_2_test_input = torch.from_numpy(np.array(class_2_test_input)).unsqueeze(0).float()

        #### Get Output
        cross_test_output = model.forward(cross_test_input)
        class_1_test_output = model.forward(class_1_test_input)
        class_2_test_output = model.forward(class_2_test_input)

        #### Get Output Matrix
        cross_matrix = model.get_output_matrix(cross_test_input, cross_test_output, pandas = True)
        class_1_matrix = model.get_output_matrix(class_1_test_input, class_1_test_output, pandas = True)
        class_2_matrix = model.get_output_matrix(class_2_test_input, class_2_test_output, pandas = True)


        #### Get D11, D22, D11*, D22*, D12*
        class_1_inner_distance = get_inner_class_distance(class_1_matrix, class_1_sample_name_list, order = order)
        class_2_inner_distance = get_inner_class_distance(class_2_matrix, class_2_sample_name_list, order = order)
        cross_class_1_inner_distance = get_inner_class_distance(cross_matrix, class_1_star_sample_name_list, order = order)
        cross_class_2_inner_distance = get_inner_class_distance(cross_matrix, class_2_star_sample_name_list, order = order)
        cross_inter_distance = get_inter_class_distance(cross_matrix, class_1_star_sample_name_list, class_2_star_sample_name_list, order = order)

        class_1_inner_distance_list.append(class_1_inner_distance)
        class_2_inner_distance_list.append(class_2_inner_distance)
        class_1_star_inner_distance_list.append(cross_class_1_inner_distance)
        class_2_star_inner_distance_list.append(cross_class_2_inner_distance)
        inter_distance_list.append(cross_inter_distance)

    #### Get Average Distance
    class_1_inner_distance = np.mean(class_1_inner_distance_list)
    class_2_inner_distance = np.mean(class_2_inner_distance_list)
    class_1_star_inner_distance = np.mean(class_1_star_inner_distance_list)
    class_2_star_inner_distance = np.mean(class_2_star_inner_distance_list)
    inter_distance = np.mean(inter_distance_list)
    return class_1_inner_distance, class_2_inner_distance, class_1_star_inner_distance, class_2_star_inner_distance, inter_distance
    

##########################################################################
lifelog_itemlist = "/home/li/datasets/lifelog/itemlist.csv"
lifelog_data = pd.read_csv(lifelog_itemlist)
group_path = "/home/li/datasets/lifelog/Group1_64.txt"
group_list = []
group_item_name_list = []

input_csv = "/home/li/torch/data/Data_Input_200_LI_Mofei_20190518.csv"
output_csv = "/home/li/torch/data/Data_Output_200_LI_Mofei_20190518.csv"





with open(group_path,"r") as g_f:
    for line in g_f.readlines():
        group_list.append(int(line.strip()))
        group_item_name_list.append(lifelog_data.loc[int(line.strip()) - 1,"Name"])

################## PARAMS


## Train Params
BATCH_SIZE = 1
LEARNING_RATE = 0.5
WEIGHT_DECAY = torch.tensor(0.00001).float()
QUERY_DIM = 5
KEY_DIM = 10
FEATURE_DIM = 8
EPOCH = 50

## Evaluation Params
CV_NUM = 5
EVA_SAMPLE_NUMBER = 30
ORDER = 2

model_path = "/home/li/torch/model/attention_net_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_CV.model"

if __name__ == '__main__':

    ############## Data Preparation ###################

    


    
