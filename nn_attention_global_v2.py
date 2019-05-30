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

BATCH_SIZE = 1
LEARNING_RATE = 0.02
WEIGHT_DECAY = torch.tensor(0.0001).float()
QUERY_DIM = 5
KEY_DIM = 10
FEATURE_DIM = 8

model_path = "/home/li/torch/model/attention_net_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + ".model"
test_csv_file = "/home/li/torch/csv/test_csv.csv"


if __name__ == '__main__':

    data_file_path = "/home/li/datasets/lifelog/data/Group1_li_mofei_no_20_20190520.csv"
    user = "li_mofei"
    
    name_list = group_item_name_list
    
    input_dim = len(name_list)

    dataset = GlobalModelDataset(input_csv, output_csv)

    data_num = dataset.data_num
    test_data_num = int(data_num/5)
    train_data_num = data_num - test_data_num

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_data_num, test_data_num))
    print(dataset.get_parameters())
    params = (QUERY_DIM, KEY_DIM, FEATURE_DIM)
    attention_net = Attention_Net(dataset,params)
    
    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            num_workers = 0)

    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = BATCH_SIZE,
                                  shuffle = True,
                                  num_workers = 0)

    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size = BATCH_SIZE,
                                 shuffle = True,
                                 num_workers = 0)
    
    print(len(dataloader))
    
    optimizer = torch.optim.Adam(attention_net.parameters(), lr = LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    for name,param in attention_net.named_parameters():
        if param.requires_grad:
            print(name)
            #print(param)


    ###################### Training ###############
    attention_net.train()

    
    print(dataloader)
    train_loss_list = []
    test_loss_list = []
    for epoach in range(20):
        train_loss_each = 0
        test_loss_each = 0

        ## Train
        for im,label in train_dataloader:
            l0_regularization = torch.tensor(0).float()
            l1_regularization = torch.tensor(0).float()
            l2_regularization = torch.tensor(0).float()
            
            out = attention_net.forward(im)
            #print(im.shape)
            #print(out.shape)
            #print(label.shape)
            
            mse_loss = loss_function(out,label)
            
            #print(loss)
            for param in attention_net.parameters():
                l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                l2_regularization += WEIGHT_DECAY * torch.norm(param,2)

            loss = mse_loss + l0_regularization
            train_loss_each += loss.item()/train_data_num
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoach: " + str(epoach) + " , Train Loss: " + str(train_loss_each))
        train_loss_list.append(train_loss_each)

        ## Test
        for im,label in test_dataloader:
            l0_regularization = torch.tensor(0).float()
            l1_regularization = torch.tensor(0).float()
            l2_regularization = torch.tensor(0).float()
            
            out = attention_net.forward(im)
            mse_loss = loss_function(out,label)
            
            for param in attention_net.parameters():
                l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                l2_regularization += WEIGHT_DECAY * torch.norm(param,2)


            loss = mse_loss + l0_regularization
            test_loss_each += loss.item()/test_data_num

        print("Epoach: " + str(epoach) + " , Test Loss: " + str(test_loss_each))
        test_loss_list.append(test_loss_each)
            
        

    torch.save(attention_net, model_path)
    
    attention_net.eval()
    #final_input = torch.from_numpy(np.ones(input_dim)).float().unsqueeze(0)
    #final_output = attention_net.forward(final_input)
    #final_matrix = attention_net.get_output_matrix(final_input, final_output,pandas = True)
    #csv_output_file = "/home/li/torch/result/test_result_" + str(user) + ".csv"
    #final_matrix.to_csv(csv_output_file)
    #print(loss_list)
    plt_file = "/home/li/torch/figure/training_curve.png"
    plt.plot(range(len(train_loss_list)), train_loss_list ,label = "train loss")
    plt.plot(range(len(test_loss_list)), test_loss_list ,label = "test loss")
    plt.legend(loc = 'upper right')
    plt.savefig(plt_file)
    plt.show()

    

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

    
