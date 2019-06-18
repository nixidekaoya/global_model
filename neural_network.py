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


##################### Attention Net Class #######################
        
class Attention_Net(nn.Module):
    def __init__(self,dataset,params = (5,10,8), activation = "sigmoid"):
        super(Attention_Net,self).__init__()
        self.dataset = dataset
        self.item_list = dataset.item_list
        self.item_number = len(self.item_list)
        self.output_dim = int((self.item_number * (self.item_number - 1))/2)

        self.query_dim = int(params[0])
        self.key_dim = int(params[1])
        self.feature_dim = int(params[2])
        self.linear_layer1 = nn.Linear(self.item_number,self.query_dim)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU(True)
        
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
        x = self.act(x)
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

        #print(x)

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

        print(inp)
        output_mask = self.get_output_mask(inp)
        output = output.mul(output_mask)
        output = list(output[0].detach().numpy())
        
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
        input_list = list(inp)
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
    
