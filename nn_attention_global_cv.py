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




with open(group_path,"r") as g_f:
    for line in g_f.readlines():
        group_list.append(int(line.strip()))
        group_item_name_list.append(lifelog_data.loc[int(line.strip()) - 1,"Name"])

################## PARAMS
## Constant
ADAM = "Adam"
SGD = "SGD"
L0 = "L0"
L1 = "L1"
L2 = "L2"
MSE = "MSE"
WD = "0001"
ATTENTION = "attention_net"
LINEAR = "linear_net"
RELU = "relu"
SIGMOID = "sigmoid"


## Train Params
NET = ATTENTION
BATCH_SIZE = 10
LEARNING_RATE = 0.05
WEIGHT_DECAY = torch.tensor(0.0001).float()
QUERY_DIM = 9
KEY_DIM = 6
FEATURE_DIM = 5
EPOCH = 30
MOMENTUM = 0.9
REG = L1
ACT = SIGMOID
OPTIMIZER = SGD

## Evaluation Params
EVA_SAMPLE_NUMBER = 30
BETAS = (0.9,0.999)
LOSS = MSE
CV_NUM = 4


if __name__ == '__main__':

    
    ############## Data Preparation ###################
    username = "artificial"

    extra = "Artificial_Data_"
    model_path = "/home/li/torch/model/" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + "_CV.model" 
    train_log_path = "/home/li/torch/model/train_log/"  + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + ".txt" 

    input_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_X_Y_input.csv"
    output_csv = "/home/li/torch/artificial_data/artificial_data_10000_class_1_4_X_Y_output.csv"
    dataset = GlobalModelDataset(input_csv, output_csv)

    data_num = dataset.data_num

    sample_data_num = int(data_num/CV_NUM)

    train_data_num = data_num - sample_data_num
    test_data_num = sample_data_num

    splits_list = []
    for i in range(CV_NUM):
        splits_list.append(sample_data_num)
    splits_list = tuple(splits_list)

    datasets = torch.utils.data.random_split(dataset, splits_list)

    dataloader_list = []
    for ds in datasets:
        dataloader = DataLoader(dataset = ds,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = 0)
        dataloader_list.append(dataloader)

    data_num = dataset.data_num

    sample_data_num = int(data_num/CV_NUM)

    params = (QUERY_DIM,KEY_DIM,FEATURE_DIM)

    ## Attention Net
    if NET == ATTENTION:
        net = Attention_Net(dataset, params, activation = ACT)
    ## Linear Net
    elif NET == LINEAR:
        net = Linear_Net(dataset, FEATURE_DIM)



    ## Optimizer
    if OPTIMIZER == SGD:
        optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    elif OPTIMIZER == ADAM:
        optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE, betas = BETAS)

    ## Loss
    loss_function = torch.nn.MSELoss()

    #### Print Parameters
    #for name,param in net.named_parameters():
    #    if param.requires_grad:
    #        print(name)
            #print(param)
    ###################### Training ############### Cross Validation
    #attention_net.train()
    #print(dataloader)
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(EPOCH):
        train_loss_each_epoch_list = []
        test_loss_each_epoch_list = []

        for i in range(CV_NUM):
            test_dataloader = dataloader_list[i]
            train_dataloader_list = dataloader_list[:i] + dataloader_list[i+1:]            

            net.train()
            #print(len(train_dataloader_list))
            ###### Train
            train_loss_each_sample_list = []
            for dataloader in train_dataloader_list:
                
                train_loss_each = 0
                for im,label in dataloader:
                    l0_regularization = torch.tensor(0).float()
                    l1_regularization = torch.tensor(0).float()
                    l2_regularization = torch.tensor(0).float()

                    if NET == ATTENTION:
                        out,dist = net.forward(im)
                    elif NET == LINEAR:
                        out = net.forward(im)
                    mse_loss = loss_function(out,label)

                    ## Regularization
                    for param in net.parameters():
                        l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                        l2_regularization += WEIGHT_DECAY * torch.norm(param,2)

                    if REG == L0:
                        loss = mse_loss + l0_regularization
                    elif REG == L1:
                        loss = mse_loss + l1_regularization
                    elif REG == L2:
                        loss = mse_loss + l2_regularization
                    
                    train_loss_each += mse_loss.item()/sample_data_num
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss_each_sample_list.append(train_loss_each)
                #print(len(train_loss_each_sample_list))
            #print(len(train_loss_each_sample_list))
            train_loss_each_epoch_list.append(np.mean(train_loss_each_sample_list))


            ############ Test
            test_loss_each = 0
            net.eval()
            for im,label in test_dataloader:

                if NET == ATTENTION:
                    out,dis = net.forward(im)
                elif NET == LINEAR:
                    out = net.forward(im)
                #out = linear_net.forward(im)
                mse_loss = loss_function(out,label)
                test_loss_each += mse_loss.item()/sample_data_num

            test_loss_each_epoch_list.append(test_loss_each)


        train_loss = np.mean(train_loss_each_epoch_list)
        test_loss = np.mean(test_loss_each_epoch_list)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        info1 = "Epoch: " + str(epoch) + " , Train Loss: " + str(train_loss)
        info2 = "Epoch: " + str(epoch) + " , Test Loss: " + str(test_loss)
        print(info1)
        print(info2)
        if NET == ATTENTION:
            info3 = "Epoch: " + str(epoch) + " , Distribution: " + str(dist)
        print(info3)
        


    torch.save(net.state_dict(), model_path)



    


    
