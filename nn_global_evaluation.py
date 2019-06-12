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
        self.distribute = x
        x = x.mm(self.value_matrix)

        #Decoder
        x = self.linear_layer2(x)
        x = x.mul(mask)

        return x,self.distribute
        


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
            output_matrix[i,j] = math.exp(output[self.dataset.from_ij_get_index(i,j)])
            output_matrix[j,i] = output_matrix[i,j]

        if pandas == False:
            return torch.from_numpy(output_matrix)
        else:
            return pd.DataFrame(output_matrix, columns = self.item_list, index = self.item_list)

    def get_output_small_matrix(self, inp, output, pandas = False):

        output_mask = self.get_output_mask(inp)
        output = output.mul(output_mask)

        output = list(output[0].detach().numpy())
        print(inp.shape)
        input_list = list(inp[0].detach().numpy())
        input_item_list = []
        index_list = []

        for i in range(self.item_number):
            if float(input_list[i]) == float(1):
                input_item_list.append(self.item_list[i])
                index_list.append(i)
                
        item_num = len(input_item_list)
        com = itertools.combinations(range(item_num),2)
        output_matrix = np.zeros([item_num, item_num], dtype = float)
        
        for c in com:
            i = c[0]
            j = c[1]
            index_i = index_list[i]
            index_j = index_list[j]
            output_matrix[i,j] = math.exp(output[self.dataset.from_ij_get_index(index_i,index_j)])
            output_matrix[j,i] = output_matrix[i,j]            

        if pandas == False:
            return torch.from_numpy(output_matrix)
        else:
            return pd.DataFrame(output_matrix, columns = input_item_list, index = input_item_list)
        

    
######## Non Attention Net Class
class Linear_Net(nn.Module):
    def __init__(self,dataset,params = 8):
        super(Linear_Net,self).__init__()
        self.dataset = dataset
        self.item_list = dataset.item_list
        self.item_number = len(self.item_list)
        self.output_dim = int((self.item_number * (self.item_number - 1))/2)

        self.feature_dim = int(params)
        self.linear_layer1 = nn.Linear(self.item_number,self.feature_dim)
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
        #print(inp.shape)
        
        output = list(output[0].detach().numpy())
        #print(len(output))
        output_matrix = np.zeros([self.item_number,self.item_number], dtype = float)
        for com in self.dataset.com_list:
            i = com[0]
            j = com[1]
            output_matrix[i,j] = math.exp(output[self.dataset.from_ij_get_index(i,j)])
            output_matrix[j,i] = output_matrix[i,j]

        if pandas == False:
            return torch.from_numpy(output_matrix)
        else:
            return pd.DataFrame(output_matrix, columns = self.item_list, index = self.item_list)

    def get_output_small_matrix(self, inp, output, pandas = False):

        output_mask = self.get_output_mask(inp)
        output = output.mul(output_mask)

        output = list(output[0].detach().numpy())

        #print(inp.shape)
        input_list = list(inp[0].detach().numpy())
        input_item_list = []
        index_list = []

        for i in range(self.item_number):
            if float(input_list[i]) == float(1):
                input_item_list.append(self.item_list[i])
                index_list.append(i)

        #print(index_list)
        #print(input_item_list)

        item_num = len(input_item_list)
        
        com = itertools.combinations(range(item_num),2)
        output_matrix = np.zeros([item_num, item_num], dtype = float)
        
        for c in com:
            i = c[0]
            j = c[1]
            index_i = index_list[i]
            index_j = index_list[j]
            output_matrix[i,j] = math.exp(output[self.dataset.from_ij_get_index(index_i,index_j)])
            output_matrix[j,i] = output_matrix[i,j]            

        if pandas == False:
            return torch.from_numpy(output_matrix)
        else:
            return pd.DataFrame(output_matrix, columns = input_item_list, index = input_item_list)
    

   
        

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

    linear_model_path = "/home/li/torch/model/linear_net_u_nakamura_F_5_CV.model"
    attention_model_path = "/home/li/torch/model/attention_net_u_nakamura_Q_9_K_3_F_5_CV.model"

    username = "nakamura"

    input_csv = "/home/li/torch/data/Data_Input_164_nakamura_20190605.csv"
    output_csv = "/home/li/torch/data/Data_Output_164_nakamura_20190605.csv"

    normal_path = "/home/li/torch/figure/normal_net/"
    attention_path = "/home/li/torch/figure/attention_net/"

    normal_plot_path = normal_path + "normal_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + "_output_mds_figure.png"
    normal_csv_path = normal_path + "normal_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + "_output_distance.csv"

    attention_plot_path = attention_path + "attention_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + "_output_mds_figure.png"
    attention_csv_path = attention_path + "attention_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + "output_distance.csv"

    normal_item_name_path = normal_path + "item_name_normal_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + ".txt"
    attention_item_name_path = attention_path + "item_name_attention_net_object_" + str(OBJECT_NUM) + "_u_" + str(username) + ".txt"


    bar_path = "/home/li/torch/figure/attention_net/distribution/bar_graph_" + str(OBJECT_NUM) + "_u_" + str(username) + ".png"

    
    linear_model = torch.load(linear_model_path)
    attention_model = torch.load(attention_model_path)
    linear_model.eval()
    attention_model.eval()

    
    dataset = GlobalModelDataset(input_csv, output_csv)
    item_list = dataset.item_list

    normal_embedding = MDS(n_components = 2, dissimilarity = "precomputed")
    attention_embedding = MDS(n_components = 2, dissimilarity = "precomputed")

    input_sample = random.sample(range(64),OBJECT_NUM)

    #input_sample = [4,14,45,62,35,22,54,23]
        
    input_name_list = []
    for i in input_sample:
        input_name_list.append(item_list[i])

    with open(normal_item_name_path, 'w') as item_f:
        for item in input_name_list:
            item_f.write(str(item) + "\r\n")

    with open(attention_item_name_path, 'w') as item_f:
        for item in input_name_list:
            item_f.write(str(item) + "\r\n")

    input_test = []
    for item in item_list:
        if item in input_name_list:
            input_test.append(1)
        else:
            input_test.append(0)

    input_test = torch.from_numpy(np.array(input_test)).unsqueeze(0).float()

    
    attention_output,dist = attention_model.forward(input_test)
    normal_output = linear_model.forward(input_test)
    
    normal_output_matrix = linear_model.get_output_small_matrix(input_test, normal_output, pandas = False)
    attention_output_matrix = attention_model.get_output_small_matrix(input_test, attention_output, pandas = False)
    
    normal_output_df = linear_model.get_output_small_matrix(input_test, normal_output, pandas = True)
    attention_output_df = attention_model.get_output_small_matrix(input_test, attention_output, pandas = True)
    

    normal_pos = normal_embedding.fit_transform(normal_output_matrix)
    attention_pos = attention_embedding.fit_transform(attention_output_matrix)

    dist = list(dist[0].detach().numpy())

    print(dist)

    #### ATTENTION PLOT
    x_list = []
    y_list = []
    for p in attention_pos:
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list, c = "red", marker = "o")
    for i in range(len(input_name_list)):
        plt.annotate(input_name_list[i], tuple(attention_pos[i]))

    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.savefig(attention_plot_path)
    
    attention_output_df.to_csv(attention_csv_path)

    plt.close('all')

    plt.bar(range(len(dist)), dist, color = 'b')
    plt.savefig(bar_path)
    plt.close('all')

    #### NORMAL PLOT
    x_list = []
    y_list = []
    for p in normal_pos:
        x_list.append(p[0])
        y_list.append(p[1])

    plt.scatter(x_list, y_list, c = "red", marker = "o")
    for i in range(len(input_name_list)):
        plt.annotate(input_name_list[i], tuple(normal_pos[i]))

    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.savefig(normal_plot_path)
    
    normal_output_df.to_csv(normal_csv_path)

    plt.close('all')

    
    
    

    

        
