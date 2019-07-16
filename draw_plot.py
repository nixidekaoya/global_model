#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import random
import itertools
import math
import os
import matplotlib.pyplot as plt





'''
grad_search_path = "/home/li/torch/grad_search/20190604/"
grad_search_result_csv_path = grad_search_path + "result.csv"
backup_path = grad_search_path + "backup.txt"
plot_path = grad_search_path + "parameters_plot.txt"

plt_file = "/home/li/torch/parameters_plot.txt"
plot_file = open(plt_file,"r")

graph_file = "/home/li/torch/parameters_plot.png"

least_c3 = 1
largest_c3 = 0
x_label = ["c1","c2","c3"]
largest_param = "(2,10,6)"
least_param = "(9,3,5)"

for line in plot_file.readlines():
    splits = line.split("\t")
    #print(splits)
    param = splits[0]
    c1 = float(splits[1])
    c2 = float(splits[2])
    c3 = float(splits[3])
    cs = [c1,c2,c3]

    if c3 > largest_c3:
        largest_c3 = c3
        largest_c3_param = param

    if c3 < least_c3:
        least_c3 = c3
        least_c3_param = param
    
    plt.plot(x_label,cs)

print("Largest:" + str(largest_c3_param))
print("Least:" + str(least_c3_param))
plt.annotate(str(largest_c3_param), xy = ("c3",largest_c3))
plt.annotate(str(least_c3_param), xy = ("c3",least_c3))
plt.savefig(graph_file)
plt.close()
'''
