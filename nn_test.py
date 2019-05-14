#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable




class Attention_Net(nn.Module):
    def __init__(self,item_list):
        super(Attention_Net,self).__init__()
        w2v_file = "/home/li/word2vec/lifelog_w2v.csv"
        self.w2v_data = pd.read_csv(w2v_file)
        self.context_vector_dim = 300
        self.item_list = item_list
        
        self.item_number = len(item_list)
        self.output_dim = int(self.item_number * (self.item_number - 1)/2)
        self.key_matrix = Variable(torch.randn(self.item_number, self.item_number), requires_grad = True)
        np1 = np.zeros((self.context_vector_dim,self.item_number), dtype = float)
        for i in range(self.item_number):
            item = str(item_list[i])
            np1[:,i] = np.array(list(self.w2v_data[item]))
        self.value_matrix = Variable(torch.from_numpy(np1).float(), requires_grad = False)
        self.linear_layer1 = nn.Linear(self.context_vector_dim, self.context_vector_dim * 9)
        self.relu_layer1 = nn.ReLU()
        self.linear_layer2 = nn.Linear(self.context_vector_dim * 9, self.context_vector_dim * 54)
        self.relu_layer2 = nn.ReLU()
        self.linear_layer3 = nn.Linear(self.context_vector_dim * 54, self.output_dim)
            

    def forward(self,x):
        #Encoder
        mask = self.get_mask(x)
        x = x.mm(self.key_matrix)
        x = F.softmax(x,dim = 1)
        x = x.mm(self.value_matrix.transpose(1,0))

        #Decoder
        x = self.linear_layer1(x)
        x = self.relu_layer1(x)
        x = self.linear_layer2(x)
        x = self.relu_layer2(x)
        x = self.linear_layer3(x)

        x = x.mul(mask)

        return x
        
        
    def count_non_zero(self,x):
        result = 0

        return result


    def get_mask(self,x):
        mask = []
        x = x.data.numpy()
        x = x[0]
        item_number = len(x)
        for i in range(1,item_number):
            for j in range(item_number - i):
                if int(x[j]) == 1 and int(x[j+i]) == 1:
                    mask.append(float(1))
                else:
                    mask.append(float(0))
        mask = Variable(torch.from_numpy(np.array(mask)).float())
        return mask
                

w2v_file = "/home/li/word2vec/lifelog_w2v.csv"

if __name__ == '__main__':
    data = pd.read_csv(w2v_file)
    name_list = list(data.columns)
    name_list = name_list[1:]
    attention_net = Attention_Net(name_list)

    
    test_input = Variable(torch.randn(1,256))
    test_output = attention_net.forward(test_input)
    print(test_output)
