#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:04:37 2022

@author: user
"""

import numpy as np

def repair2(individual2,dim,qubits_num5,pick_list1,unique_val,counts,double_val_ind):
    browse_pick = 0
    individual3 = np.copy(individual2)
    for indx6 in range(len(double_val_ind)):
         double_val = unique_val[double_val_ind[indx6]]
         repeat = counts[double_val_ind[indx6]]
         indx7 = 0
         while (indx7 < dim) and (repeat > 1):
             if (individual3[indx7] == double_val) & (repeat > 1):
                 individual3[indx7] = pick_list1[browse_pick]
                 browse_pick = browse_pick + 1               
                 repeat = repeat - 1 
             indx7 =  indx7 + 1
    return individual3; 

def main():
    qubits_num0 = 20
    dim = 11
    pop = np.array([1,2,2,3,3,4,4,19,3,8,0,20])
    unique_val, counts = np.unique(pop, return_counts=True)
    double_val_ind = np.where(counts>1)
    double_val_ind = np.copy(double_val_ind[0])
    # redundant_val = unique_val[double_val_ind]
    while len(double_val_ind) > 0:
        # pickup list 
        pick_list= []
        for indx10 in range(qubits_num0):
            indx9 = 0;
            check_val = False
            while (indx9 < len(unique_val)) & (check_val == False):
                if unique_val[indx9] == indx10 :
                    check_val = True
                indx9 = indx9 + 1
            if  check_val == False:
                pick_list.append(indx10)
        pop= repair2(pop,dim,qubits_num0,pick_list,unique_val,counts,double_val_ind)
        unique_val, counts = np.unique(pop, return_counts=True)
        double_val_ind = np.where(counts>1)
        double_val_ind = np.copy(double_val_ind[0])
    return;
if __name__ == '__main__':
    main();