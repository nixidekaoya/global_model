#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import random
import itertools
import math
import time
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

################## PARAMS

## Const
ADAM = "Adam"
SGD = "SGD"
L0 = "L0"
L1 = "L1"
L2 = "L2"
MSE = "MSE"
ATTENTION = "attention_net"
LINEAR = "linear_net"
RELU = "relu"
SIGMOID = "sigmoid"

## Train Params
OPT = SGD
WD = 0.00001
BATCH_SIZE = 10
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = torch.tensor(WD).float()
QUERY_DIM = 9
KEY_DIM = 5
FEATURE_DIM = 6
EPOCH = 30
BETAS = (0.9,0.999)
REG = L1
LOSS = MSE
ACT = SIGMOID
CV_NUM = 5


## Evaluation Params
EVA_SAMPLE_NUMBER = 30
ORDER = 1

## Grad Search Params
Q_RANGE = (9,10)
K_RANGE = (6,7)
F_RANGE = (5,6)



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

def evaluate_model_inner_inter_distance(model, sample_number = 10, combines = (4,4), order = 1):
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
        cross_test_output,dis = model.forward(cross_test_input)
        class_1_test_output,dis = model.forward(class_1_test_input)
        class_2_test_output,dis = model.forward(class_2_test_input)

        #### Get Output Matrix
        cross_matrix = model.get_output_matrix(cross_test_input, cross_test_output, pandas = True)
        class_1_matrix = model.get_output_matrix(class_1_test_input, class_1_test_output, pandas = True)
        class_2_matrix = model.get_output_matrix(class_2_test_input, class_2_test_output, pandas = True)


        #### Get D11, D22, D11*, D22*, D12*
        class_1_inner_distance = get_inner_class_distance(class_1_matrix, class_1_star_sample_name_list, order = order)
        class_2_inner_distance = get_inner_class_distance(class_2_matrix, class_2_star_sample_name_list, order = order)
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
    

def grad_search(dataset, data_loader_list, grad_search_path, params = (1,1,1)):

    QUERY_DIM = params[0]
    KEY_DIM = params[1]
    FEATURE_DIM = params[2]

    
    ############## Data Preparation ###################

    model_path = "/home/li/torch/model/attention_net_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_CV.model"
    this_search_path = grad_search_path + "Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "/"


    if not os.path.exists(this_search_path):
        os.mkdir(this_search_path)


    data_file_path = "/home/li/datasets/lifelog/data/Group1_li_mofei_no_20_20190520.csv"
    user = "li_mofei"

    
    train_log_file_path = this_search_path + "train_log.txt"    

    name_list = dataset.item_list
    input_dim = len(name_list)

    data_num = dataset.data_num

    sample_data_num = int(data_num/CV_NUM)

    ## Attention Net
    attention_net = Attention_Net(dataset, params, activation = ACT)


    ## Optimizer
    if OPT == SGD:
        optimizer = torch.optim.SGD(attention_net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    elif OPT == ADAM:
        optimizer = torch.optim.Adam(attention_net.parameters(), lr = LEARNING_RATE, betas = BETAS)
        

    ## Train Log
    with open(train_log_file_path, 'w') as train_f:
        train_f.write("Optimizer: " + OPT + "\r\n")
        train_f.write("Learning Rate: " + str(LEARNING_RATE) + "\r\n")
        if OPT == SGD:
            train_f.write("Momentum: " + str(MOMENTUM) + "\r\n")
        elif OPT == ADAM:
            train_f.write("Betas: " + str(BETAS) + "\r\n")
        train_f.write("Regularization: " + str(REG) + "\r\n")
        train_f.write("Loss: " + str(LOSS) + "\r\n")
        train_f.write("Parameters: \r\n" )
        for name,param in attention_net.named_parameters():
            if param.requires_grad:
                train_f.write(str(name) + "\r\n")

    train_log_file = open(train_log_file_path, 'a')
            
    ## Loss
    if LOSS == MSE:
        loss_function = torch.nn.MSELoss()

    #### Print Parameters
    #for name,param in attention_net.named_parameters():
    #    if param.requires_grad:
    #        print(name)
            #print(param)


    ###################### Training ############### Cross Validation
    #attention_net.train()

    
    #print(dataloader)
    train_loss_list = []
    test_loss_list = []

    class_1_distance_list = []
    class_2_distance_list = []
    class_1_star_distance_list = []
    class_2_star_distance_list = []
    inter_distance_list = []
    coeff_1_list = []
    coeff_2_list = []
    coeff_3_list = []
    
    for epoch in range(EPOCH):
        train_loss_each_epoch_list = []
        test_loss_each_epoch_list = []

        for i in range(CV_NUM):
            test_dataloader = dataloader_list[i]
            train_dataloader_list = dataloader_list[:i] + dataloader_list[i+1:]            
            attention_net.train()
            #print(len(train_dataloader_list))
            ###### Train
            train_loss_each_sample_list = []
            for dataloader in train_dataloader_list:
                
                train_loss_each = 0
                for im,label in dataloader:
                    l0_regularization = torch.tensor(0).float()
                    l1_regularization = torch.tensor(0).float()
                    l2_regularization = torch.tensor(0).float()

                    out,dis = attention_net.forward(im)
                    mse_loss = loss_function(out,label)

                    ## Regularization
                    for param in attention_net.parameters():
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
            attention_net.eval()
            for im,label in test_dataloader:
                out,dis = attention_net.forward(im)
                mse_loss = loss_function(out,label)
                test_loss_each += mse_loss.item()/sample_data_num

            test_loss_each_epoch_list.append(test_loss_each)

        train_loss = np.mean(train_loss_each_epoch_list)
        test_loss = np.mean(test_loss_each_epoch_list)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)



        ######################## Evaluation
        info1 = "Epoch: " + str(epoch) + " , Train Loss: " + str(train_loss)
        info2 = "Epoch: " + str(epoch) + " , Test Loss: " + str(test_loss)

        class_1_distance, class_2_distance , class_1_star_distance, class_2_star_distance, inter_distance = evaluate_model_inner_inter_distance(attention_net, sample_number = EVA_SAMPLE_NUMBER, order = ORDER)
        class_1_distance_list.append(class_1_distance)
        class_2_distance_list.append(class_2_distance)
        class_1_star_distance_list.append(class_1_star_distance)
        class_2_star_distance_list.append(class_2_star_distance)
        inter_distance_list.append(inter_distance)
        coeff_1 = float(class_1_star_distance / class_1_distance)
        coeff_2 = float(class_2_star_distance / class_2_distance)
        coeff_3 = float((class_1_star_distance + class_2_star_distance) / (2 * inter_distance))
        coeff_1_list.append(coeff_1)
        coeff_2_list.append(coeff_2)
        coeff_3_list.append(coeff_3)
        

        info3 = "Epoch: " + str(epoch) + " , D11: " + str(class_1_distance) + " , D22: " + str(class_2_distance)
        info4 = "Epoch: " + str(epoch) + " , D11*: " + str(class_1_star_distance) + ", D22*: " + str(class_2_star_distance) + ", D12*: " + str(inter_distance)
        info5 = "Epoch: " + str(epoch) + " , Coefficient 1: " + str(coeff_1) + " , Coefficient 2: " + str(coeff_2) + " , Coefficient 3: " + str(coeff_3)

        train_log_file.write(info1 + "\r\n")
        train_log_file.write(info2 + "\r\n")
        train_log_file.write(info3 + "\r\n")
        train_log_file.write(info4 + "\r\n")
        train_log_file.write(info5 + "\r\n")

    #torch.save(attention_net, model_path)
    ##################################### Get Output
    #attention_net.eval()
    #final_input = torch.from_numpy(np.ones(input_dim)).float().unsqueeze(0)
    #final_output = attention_net.forward(final_input)
    #final_matrix = attention_net.get_output_matrix(final_input, final_output,pandas = True)
    #csv_output_file = "/home/li/torch/result/test_result_" + str(user) + ".csv"
    #final_matrix.to_csv(csv_output_file)
    #print(loss_list)


    ############################### Get Training Curve
    extra = "_Optim_SGD_WD_L1_lr_05_epoach_50_"
    figure = "train_curve"
    plt_file = this_search_path + "Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + str(extra) + str(figure) + ".png"
    plt.plot(range(len(train_loss_list)), train_loss_list ,label = "train loss")
    plt.plot(range(len(test_loss_list)), test_loss_list ,label = "test loss")
    plt.legend(loc = 'upper right')
    #plt.ylim(bottom = 0, top = 0.02)
    plt.savefig(plt_file)
    #plt.show()
    plt.close('all')

    ############################ Get Distance Curve
    figure = "distance_curve"
    plt_file = this_search_path + "Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + str(extra) + str(figure) + ".png"
    
    plt.plot(range(len(class_1_distance_list)), class_1_distance_list, label = "D11")
    plt.plot(range(len(class_2_distance_list)), class_2_distance_list, label = "D22")
    plt.plot(range(len(class_1_star_distance_list)), class_1_star_distance_list, label = "D11*")
    plt.plot(range(len(class_2_star_distance_list)), class_2_star_distance_list, label = "D22*")
    plt.plot(range(len(inter_distance_list)), inter_distance_list, label = "D12*")
    plt.legend(loc = 'upper right')
    plt.savefig(plt_file)
    #plt.show()
    plt.close('all')

    ############################ Get Coeff Curve
    figure = "coefficient_curve"
    plt_file = this_search_path + "Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + str(extra) + str(figure) + ".png"
    
    plt.plot(range(len(coeff_1_list)), coeff_1_list, label = "C1")
    plt.plot(range(len(coeff_2_list)), coeff_2_list, label = "C2")
    plt.plot(range(len(coeff_3_list)), coeff_3_list, label = "C3")
    plt.legend(loc = 'upper right')
    plt.savefig(plt_file)
    #plt.show()
    plt.close('all')

    ############################ Get Coeff Scatter Plot Versus Loss
    figure = "coefficient_scatter_plot"
    plt_file = this_search_path + "Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + str(extra) + str(figure) + ".png"
    
    plt.scatter(train_loss_list, coeff_1_list, c = "blue", marker = "o", label = "C1")
    plt.scatter(train_loss_list, coeff_2_list, c = "green", marker = "X", label = "C2")
    plt.scatter(train_loss_list, coeff_3_list, c = "red", marker = "v", label = "C3")
    plt.legend(loc = 'upper right')
    plt.savefig(plt_file)
    #plt.show()
    plt.close('all')

    return_tuple = (train_loss_list[-1], test_loss_list[-1], class_1_distance_list[-1], class_2_distance_list[-1], class_1_star_distance_list[-1], class_2_star_distance_list[-1], inter_distance_list[-1], coeff_1_list[-1], coeff_2_list[-1], coeff_3_list[-1])

    return return_tuple
    


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


if __name__ == '__main__':
    grad_search_path = "/home/li/torch/grad_search/20190626/"
    grad_search_result_csv_path = grad_search_path + "result.csv"
    backup_path = grad_search_path + "backup.txt"
    plot_path = grad_search_path + "parameters_plot.txt"

    input_csv = "/home/li/torch/artificial_data/artificial_data_2000_class_1_4_XoY_XoZ_input.csv"
    output_csv = "/home/li/torch/artificial_data/artificial_data_2000_class_1_4_XoY_XoZ_output.csv"
    
    if not os.path.exists(grad_search_path):
        os.mkdir(grad_search_path)

    plot_log_file = open(plot_path,'w')
    plot_log_file.close()
    
    dataset = GlobalModelDataset(input_csv, output_csv)
    data_num = dataset.data_num
    print(data_num)

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

    
    result = pd.DataFrame()
    combine_list = []
    train_loss_list = []
    test_loss_list = []
    class_1_distance_list = []
    class_2_distance_list = []
    class_1_star_distance_list = []
    class_2_star_distance_list = []
    inter_class_distance_list = []
    coeff_1_list = []
    coeff_2_list = []
    coeff_3_list = []

    t1 = time.time()

    
    for q in range(Q_RANGE[0], Q_RANGE[1] + 1):
        for k in range(K_RANGE[0], K_RANGE[1] + 1):
            for f in range(F_RANGE[0], F_RANGE[1] + 1):
                params = (q,k,f)
                combine_list.append(params)
                return_tuple = grad_search(dataset, dataloader_list, grad_search_path, params = params)
                train_loss_list.append(float(return_tuple[0]))
                test_loss_list.append(float(return_tuple[1]))
                class_1_distance_list.append(float(return_tuple[2]))
                class_2_distance_list.append(float(return_tuple[3]))
                class_1_star_distance_list.append(float(return_tuple[4]))
                class_2_star_distance_list.append(float(return_tuple[5]))
                inter_class_distance_list.append(float(return_tuple[6]))
                coeff_1_list.append(float(return_tuple[7]))
                coeff_2_list.append(float(return_tuple[8]))
                coeff_3_list.append(float(return_tuple[9]))

                print("Combines: (Q,K,F) = " + str(params))
                print("Result: " + str(return_tuple))
                ## For plotting
                with open(plot_path,'a') as plot_log_file:
                    plot_log_file.write(str(params) + "\t" + str(float(return_tuple[7])) + "\t" + str(float(return_tuple[8])) + "\t" + str(float(return_tuple[9])) + "\r\n")
                ## BACK_UP

                backup_dic = {"Combination": str(combine_list),
                              "TrainLoss": str(train_loss_list),
                              "TestLoss": str(test_loss_list),
                              "Class1Distance": str(class_1_distance_list),
                              "Class2Distance": str(class_2_distance_list),
                              "Class1StarDistance": str(class_1_star_distance_list),
                              "Class2StarDistance": str(class_2_star_distance_list),
                              "InterClassDistance": str(inter_class_distance_list),
                              "Coeff1": str(coeff_1_list),
                              "Coeff2": str(coeff_2_list),
                              "Coeff3": str(coeff_3_list)}
                    

                with open(backup_path, "w") as backup_f:
                    backup_f.write(str(backup_dic))

    
    result["(Q,K,V)"] = combine_list
    result["Train Loss"] = train_loss_list
    result["Test Loss"] = test_loss_list
    result["D11"] = class_1_distance_list
    result["D22"] = class_2_distance_list
    result["D11*"] = class_1_star_distance_list
    result["D22*"] = class_2_star_distance_list
    result["D12*"] = inter_class_distance_list
    result["C1"] = coeff_1_list
    result["C2"] = coeff_2_list
    result["C3"] = coeff_3_list

    result.to_csv(grad_search_result_csv_path)

    t2 = time.time() - t1
    print(t2)
                    
