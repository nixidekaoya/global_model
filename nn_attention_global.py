#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import random
import itertools
import math
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


############### Dataset Class #####################
class GlobalModelDataset(Dataset):
    def __init__(self,csv_path,item_list):
        self.csv_data = pd.read_csv(csv_path)
        self.item_list = item_list
        self.item_index_dic = {}
        
        self.to_tensor = transforms.ToTensor()
        self.item_number = len(self.item_list)
        self.item_each = 8
        self.data_num = int(len(self.csv_data)/self.item_each)
        self.output_dim = int((self.item_number * (self.item_number - 1))/2)
        self.input_list = []
        self.output_list = []

        for i in range(self.item_number):
            self.item_index_dic[str(self.item_list[i])] = int(i)

        for i in range(1, self.data_num + 1):
            data = self.csv_data.loc[self.csv_data["RecordID"] == i]
            input_array = np.zeros(self.item_number)
            output_matrix = np.zeros([self.item_number,self.item_number], dtype = float)

            item_name_list = []
            Xcoordinate_list = []
            Ycoordinate_list = []
            for index,row in data.iterrows():
                item_name_list.append(str(row["ItemName"]))
                Xcoordinate_list.append(float(row["X-Coordinate"]))
                Ycoordinate_list.append(float(row["Y-Coordinate"]))


            for i in range(len(item_name_list)):
                input_array[self.item_index_dic[str(item_name_list[i])]] = 1
                for j in range(1,len(item_name_list) - i):
                    c1 = (Xcoordinate_list[i], Ycoordinate_list[i])
                    c2 = (Xcoordinate_list[i+j], Ycoordinate_list[i+j])
                    output_matrix[self.item_index_dic[str(item_name_list[i])], self.item_index_dic[str(item_name_list[i+j])]] = float(self.get_distance(c1,c2))
                    output_matrix[self.item_index_dic[str(item_name_list[i+j])], self.item_index_dic[str(item_name_list[i])]] = output_matrix[self.item_index_dic[str(item_name_list[i])],self.item_index_dic[str(item_name_list[i+j])]]

            output_array = []
            for i in range(1,self.item_number):
                for j in range(self.item_number - i):
                    output_array.append(output_matrix[j,j+i])

            self.input_list.append(np.array(input_array))
            self.output_list.append(np.array(output_array))
                    
                

    def __getitem__(self,index):
        input_item = torch.from_numpy(self.input_list[index]).float()
        output_item = torch.from_numpy(self.output_list[index]).float()

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


        

##################### Attention Net Class #######################
class Attention_Net(nn.Module):
    def __init__(self,item_list):
        super(Attention_Net,self).__init__()
        self.item_list = item_list
        self.item_number = len(item_list)
        self.output_dim = int((self.item_number * (self.item_number - 1))/2)
        self.query_dim = 5
        self.key_dim = 10
        self.feature_dim = 8
        self.linear_layer1 = nn.Linear(self.item_number,self.query_dim)
        self.key_matrix = torch.nn.Parameter(torch.randn(self.query_dim,self.key_dim))
        self.value_matrix = torch.nn.Parameter(torch.randn(self.key_dim,self.feature_dim))
        self.linear_layer2 = nn.Linear(self.feature_dim, self.output_dim)
        

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
            for i in range(1,self.item_number):
                for j in range(self.item_number - i):
                    if int(batch[j]) == 1 and int(batch[j+i]) == 1:
                        sub_mask.append(float(1))
                    else:
                        sub_mask.append(float(0))
            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        #print(mask.shape)
        return mask

    def get_output_matrix(self, output):
        
        output = list(output[0].detach().numpy())
        #print(len(output))
        output_matrix = np.zeros([self.item_number,self.item_number], dtype = float)
        for i in range(1, self.item_number):
            for j in range(self.item_number - i):
                output_matrix[j,j+i] = float(output.pop(0))
                output_matrix[i+j,j] = output_matrix[j,j+i]
        return torch.from_numpy(output_matrix)


                
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

BATCH_SIZE = 5
LEARNING_RATE = 0.05



if __name__ == '__main__':

    data_file_path = "/home/li/datasets/lifelog/data/Group1_li_mofei_no_50_20190512.csv"

    
    name_list = group_item_name_list
    attention_net = Attention_Net(name_list)
    input_dim = len(name_list)

    dataset = GlobalModelDataset(data_file_path, name_list)
    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            num_workers = 0)
    optimizer = torch.optim.SGD(attention_net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    loss_function = torch.nn.MSELoss()

    for name,param in attention_net.named_parameters():
        if param.requires_grad:
            print(name)
            #print(param)


    ###################### Training ###############

    print(dataloader)
    loss = 0
    for epoach in range(50):
        for im,label in dataloader:
            out = attention_net.forward(im)
            #print(out.shape)
            #print(label.shape)
            loss = loss_function(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss.item())

    

    '''test_input = torch.zeros(input_dim)

    sample = random.sample(range(input_dim),8)
    for i in sample:
        test_input[i] = 1

    test_input = test_input.unsqueeze(0)
    print(test_input.shape)
    
    test_output = attention_net.forward(test_input)
    test_matrix = attention_net.get_output_matrix(test_output)

    print(test_output.shape)
    print(test_matrix.shape)
    print(test_matrix[sample[4],sample[7]])

    hot_vector, distance = dataset.__getitem__(3)
    print(hot_vector.shape)
    print(distance.shape)
    
    
    for name,param in attention_net.named_parameters():
        if param.requires_grad:
            print(name)
            #print(param)'''

    
