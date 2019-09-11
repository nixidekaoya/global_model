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

from neural_network import Attention_Net
from neural_network import Linear_Net
from datasets import GlobalModelDataset

import sklearn
from sklearn.decomposition import PCA


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


def get_entropy(dist_list):
    data_number = len(dist_list)
    data_dimension = len(dist_list[0])
    prob_list = []
    entropy = 0
    for i in range(data_dimension):
        prob_list.append(float(0))
    for dist in dist_list:
        prob_list[dist.index(max(dist))] += 1
    for i in range(data_dimension):
        prob_list[i] = float(prob_list[i])/data_number
    for prob in prob_list:
        if prob != float(0):
            entropy += - prob * math.log(prob)
    return entropy


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
WD = "000001"
ATTENTION = "attention_net"
LINEAR = "linear_net"
RELU = "relu"
SIGMOID = "sigmoid"


## Train Params
NET = ATTENTION
BATCH_SIZE = 10
LEARNING_RATE = 0.05
WEIGHT_DECAY = torch.tensor(0.000001).float()
QUERY_DIM = 9
KEY_DIM = 6
FEATURE_DIM = 5
EPOCH = 10000
MOMENTUM = 0.9
REG = L0
ACT = SIGMOID
OPTIMIZER = SGD

## Evaluation Params
EVA_SAMPLE_NUMBER = 30
BETAS = (0.9,0.999)
LOSS = MSE
CV_NUM = 2

TEST_NUMBER = 100


if __name__ == '__main__':
    ############## Data Preparation ###################
    username = "artificial"

    #extra = "LI_Mofei_Data_200_R_1_NL_00_LogF_True_epoch_" + str(EPOCH)
    extra = "Artificial_Data_LogF_True_epoch_" + str(EPOCH)
    model_path = "/home/li/torch/model/" + str(extra) + "_" +  str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + "_CV.model" 
    train_log_path = "/home/li/torch/model/train_log/"  + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + ".txt" 

    #input_csv = "/home/li/torch/data/Data_Input_200_LI_Mofei_20190518.csv"
    #output_csv = "/home/li/torch/data/Data_Output_200_LI_Mofei_20190518.csv"

    input_csv = "/home/li/torch/artificial_data/artificial_data_200_20190911_input.csv"
    output_csv = "/home/li/torch/artificial_data/artificial_data_200_20190911_output.csv"
    dataset = GlobalModelDataset(input_csv, output_csv, log_function = True)

    print(dataset.data_num)
    plot_path = "/home/li/torch/plot/20190911/"

    #eva_extra = "LogF_True_NL_00_li_mofei"
    eva_extra = "LogF_True_Artificial_epoch_" + str(EPOCH)
    evaluation_path = "/home/li/torch/evaluation/datanumber_600_K_" + str(KEY_DIM) + "_" + str(REG) + str(eva_extra) + "/"
    #coeff_path = "/home/li/torch/artificial_data/coefficient_logF_True_LI_Mofei_200_NL_00_test_" + str(NET) + "_epoch_" + str(EPOCH) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_WD_" + str(WD) + ".txt"
    coeff_path = "/home/li/torch/artificial_data/coefficient_logF_True_" + str(NET) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_WD_" + str(WD) + "_EPOCH_" + str(EPOCH) + ".txt"

    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    data_num = dataset.data_num
    sample_data_num = int(data_num/CV_NUM)

    if CV_NUM == 1:
        train_data_num = sample_data_num
        test_data_num = 0
    else:
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
    test_loss_log_list = []

    entropy_list = []
    
    for epoch in range(EPOCH):
        train_loss_each_epoch_list = []
        test_loss_each_epoch_list = []

        dist_list = []

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
                    dist = list(dis[0].detach().numpy())
                    dist_list.append(dist)
                elif NET == LINEAR:
                    out = net.forward(im)
                #out = linear_net.forward(im)
                mse_loss = loss_function(out,label)
                test_loss_each += mse_loss.item()/sample_data_num

            test_loss_each_epoch_list.append(test_loss_each)

        entropy = get_entropy(dist_list)
        entropy_list.append(entropy)

        train_loss = np.mean(train_loss_each_epoch_list)
        test_loss = np.mean(test_loss_each_epoch_list)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_loss_log_list.append(math.log(test_loss))

        info1 = "Epoch: " + str(epoch) + " , Train Loss: " + str(train_loss)
        info2 = "Epoch: " + str(epoch) + " , Test Loss: " + str(test_loss)
        print(info1)
        print(info2)
        if NET == ATTENTION:
            info3 = "Epoch: " + str(epoch) + " , Distribution: " + str(dis)
        print(info3)
        
        

    print(model_path)
    torch.save(net.state_dict(), model_path)

    model = net

    #### PLOT
    figure = "Learning_Curve" 
    plt_file = plot_path + str(figure) + "_" + str(extra) + "_" +  str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(test_loss_log_list)), test_loss_log_list, label = "log train loss")
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')

    figure = "Entropy_Curve"
    plt_file = plot_path + str(figure) + "_" + str(extra) + "_" +  str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + ".png"
    plt.scatter(test_loss_log_list, entropy_list, label = "Entropy")
    #plt.xlim((0,0.005))
    plt.ylim((0,1))
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')


    ##### Test
    item_list = model.item_list

    class_1_list = item_list[:32]
    class_2_list = item_list[32:]

    print(item_list)
    
    #embedding = MDS(n_components = 2, dissimilarity = "precomputed")

    d11_list = []
    d22_list = []
    d11_star_list = []
    d22_star_list = []
    d12_star_list = []

    dist_list = []
    

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
        dist_list.append(dist)

        csv_path = evaluation_path + "group1_test" + str(i) + ".csv"
        if i % 10 == 0:
            output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group1_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        if i % 10 == 0:
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
        dist_list.append(dist)

        csv_path = evaluation_path + "group2_test" + str(i) + ".csv"
        if i % 10 == 0:
            output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group2_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        if i % 10 == 0:
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
        dist_list.append(dist)

        csv_path = evaluation_path + "group3_test" + str(i) + ".csv"
        if i % 10 == 0:
            output_df.to_csv(csv_path)

        bar_path = evaluation_path + "group3_test_bar" + str(i) + ".png"
        plt.bar(range(len(dist)), dist, color = 'b')
        if i % 10 == 0:
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
    info4 = "Entropy: " + str(entropy)

    ############## PCA
    pca = PCA(n_components = "mle")
    pca.fit(dist_list)
    feature = pca.transform(dist_list)
    print(pca.explained_variance_ratio_)
    
    figure = "PCA_Test"
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    plt.scatter(feature[:,0], feature[:,1])
    plt.grid()
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.savefig(plt_file)
    plt.close('all')

 
    with open(coeff_path, "w") as log_f:
        log_f.write(info0 + "\r\n")
        log_f.write(info01 + "\r\n")
        log_f.write(info02 + "\r\n")
        log_f.write(info03 + "\r\n")
        log_f.write(info1 + "\r\n")
        log_f.write(info2 + "\r\n")
        log_f.write(info3 + "\r\n")
        log_f.write(info4 + "\r\n")
        log_f.write("Variance Ratio:" + str(pca.explained_variance_ratio_) + "\r\n")


    
