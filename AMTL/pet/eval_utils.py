import csv
import json
import pickle
import copy
import os
from typing import List, Dict, Optional
from sklearn.metrics.cluster import pair_confusion_matrix
import torch.nn.functional as F
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm

from scipy.optimize import linear_sum_assignment

def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta


def eval(gt_labels,pred_labels):
    '''
        Input:
            ground_truth_label
            predict_cluster_id
        Output:
            NMI
            
    '''
    acc=accuracy(gt_labels,pred_labels)
    ri, ari, f_beta = get_rand_index_and_f_measure(gt_labels, pred_labels, beta=1.)
    
    return acc, ri, ari, f_beta
def preprocess4Hungarian(gt_labels,pred_labels):
    gt_labels=np.array(gt_labels)
    pred_labels=np.array(pred_labels)
    
    labels_list=list(set(gt_labels))
    clusters_list=list(set(pred_labels))
    n= len(clusters_list)
    
    init_array=np.zeros([n,n])
    for i,cluster_val in enumerate(clusters_list):
        indexes= np.where(pred_labels == cluster_val)
        indexes_labels = gt_labels[indexes]
        # print("cluster_val:",cluster_val)
        # print("indexes_labels :",indexes_labels)
        for j, label_val in enumerate(clusters_list):
            # print('label_val',np.where(indexes_labels == label_val))
            count = len(np.where(indexes_labels == label_val)[0]) 
            # print(count)
            init_array[i][j]=count
    return init_array
            
    


def preprocess_for_hungarian_for_unlimited_clusters(gt_labels,pred_labels,new_class_index_start,new_class_index_end): 
    
    gt_labels=np.array(gt_labels)
    pred_labels=np.array(pred_labels)
    all_labels_list=list(set(gt_labels))
    
    labels_list=[i for i in range(new_class_index_start,new_class_index_end)]
    clusters_list=list(set(pred_labels))
   
    m= len(clusters_list)
    n= len(labels_list)
    q = len(all_labels_list)
    
    NMI_pred_labels=[]
    NMI_gt_labels=[]
    
    init_array=np.zeros([m,n])
    cluster_cocurrency_all_labels_array=np.zeros([m,q])
    
    cluster_dict={}
    label_dict={}
    for i,cluster_val in enumerate(clusters_list):
        cluster_dict[i]=cluster_val
    for j,label_val in enumerate(labels_list):
        label_dict[j]=label_val
    
    for i,cluster_val in enumerate(clusters_list):
        
        indexes= np.where(pred_labels == cluster_val)
        indexes_labels = gt_labels[indexes]
        
        NMI_pred_labels.append([ cluster_val for i in range(len(indexes[0])) ] )
        NMI_gt_labels.append(list(indexes_labels))
        
        for j, label_val in enumerate(labels_list):
            count = len(np.where(indexes_labels == label_val)[0]) 
            init_array[i][j]=count
        
        for j, label_val in enumerate(all_labels_list):
            count = len(np.where(indexes_labels == label_val)[0]) 
            # init_array[i][j]=count
            cluster_cocurrency_all_labels_array[i][j]=count
        
        
    return init_array,NMI_pred_labels,NMI_gt_labels,cluster_cocurrency_all_labels_array,cluster_dict,label_dict
    


def process_2lists(gt_labels,pred_labels):
    pass

def cluster_eval(labels=[1,1,2,2,3,3,3,3],pred_labels_list=[1,3,2,2,1,1,1,1],new_class_index_start=2,new_class_index_end=4):

    # eval
    purity, ri, ari, f_beta = eval(labels,pred_labels_list)
    print("=========evaluation=============")
    print("**purity   : ",purity)
    print("ri       : ",ri)
    print("**ari      : ",ari)
    print("**f_score  : ",f_beta)
    print("novel_NMI: ",metrics.normalized_mutual_info_score(labels,pred_labels_list))


    from scipy.optimize import linear_sum_assignment
    preprocessed_matrix,NMI_pred_labels,NMI_gt_labels,cluster_cocurrency_all_labels_array,cluster_dict,label_dict = preprocess_for_hungarian_for_unlimited_clusters(labels,pred_labels_list,new_class_index_start,new_class_index_end)
    row_ind,col_ind=linear_sum_assignment(preprocessed_matrix,True)
    
    H_assign_NMI_pred_labels=[]
    H_assign_NMI_gt_labels=[]
    for row_index in row_ind:
        H_assign_NMI_pred_labels.extend(NMI_pred_labels[row_index])
        H_assign_NMI_gt_labels.extend(NMI_gt_labels[row_index])
    print("***{} clusters H_assign_NMI: {} ".format(len(row_ind),metrics.normalized_mutual_info_score(H_assign_NMI_gt_labels,H_assign_NMI_pred_labels)))
    # H_notassign_NMI
    H_notassign_NMI_pred_labels=[]
    H_notassign_NMI_gt_labels=[]
    
    row_ind_set=set(row_ind)
    rows_count=0
    for row_index in range(len(preprocessed_matrix)): 
        if row_index not in row_ind_set:
            rows_count+=1
            H_notassign_NMI_pred_labels.extend(NMI_pred_labels[row_index])
            H_notassign_NMI_gt_labels.extend(NMI_gt_labels[row_index])
    print("{} clusters H_notassign_NMI: {} ".format(rows_count,metrics.normalized_mutual_info_score(H_notassign_NMI_gt_labels,H_notassign_NMI_pred_labels)))
    
    
    # print(preprocessed_matrix)
    print(row_ind,col_ind)
    print(preprocessed_matrix[row_ind,col_ind])
    sum_novel_sampels=preprocessed_matrix[row_ind,col_ind].sum()
    
    rows_sum = np.sum(cluster_cocurrency_all_labels_array,axis=1)
    fenmu=0
    for row in row_ind :
        fenmu += rows_sum[row] 
        
    # cluster_dict,label_dict
    cluster_match_dict={} 
    for i,row_id in enumerate(row_ind):
        key = cluster_dict[row_id]
        val = label_dict[col_ind[i]]
        cluster_match_dict[key]=val
    
    print("**novel class acc after optimal assignment Hacc: {}/{}={} ".format(sum_novel_sampels,fenmu,sum_novel_sampels/fenmu if fenmu!=0 else 0))
    print("novel class acc after optimal assignment HTotal_acc: {}/{}={} ".format(sum_novel_sampels,len(labels),sum_novel_sampels/len(labels) if len(labels)!=0 else 0 ))
    print("novel class acc after optimal assignment HTotal_hits: {}/{}={} ".format(fenmu,len(labels),fenmu/len(labels) if len(labels)!=0 else 0 ))
    
    return cluster_match_dict

import spacy
nlp=spacy.load("en_core_web_md")
def seamantic_eval(target_word_list, pred_word_list, top_num=1):
    counts= len(target_word_list) * top_num
    sim_total = 0.0
    for target_word in target_word_list :
        target_semantic = nlp(target_word) 
        for pred_word_pair in pred_word_list[:top_num]:
            pred_word = pred_word_pair[0].strip()
            pred_semantic = nlp(pred_word)
            sim_total += target_semantic.similarity(pred_semantic)
    return sim_total/counts

def seamantic_eval_sentence(target_word_list, pred_word_list, top_num=1):
    target_sentence=" ".join(target_word_list)
    pred_list = []
    for pred_word_pair in pred_word_list[:top_num]:
        pred_word = pred_word_pair[0].strip() 
        pred_list.append(pred_word)
    pred_sentence = " ".join(pred_list)
    
    target_semantic= nlp(target_sentence)       
    pred_semantic = nlp(pred_sentence)
    print(target_sentence)
    print(pred_sentence)
    res = target_semantic.similarity(pred_semantic)
    
    return res

# main()
# cluster_eval()

# pred_labels=[9 for i in range(108)]
# labels=[9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 2, 9, 0, 9, 9, 9, 3, 9, 9, 9, 9, 9,
#         9, 9, 6, 9, 9, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 7, 9, 9, 3, 0,
#         3, 9, 9, 9, 9, 9, 9, 9, 9, 3, 5, 9, 9, 9, 7, 9, 0, 9, 9, 9, 9, 9, 6, 9,
#         3, 3, 9, 9, 9, 9, 9, 9, 0, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 0, 9, 9, 9,
#         9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
# cluster_eval(labels,pred_labels,9,10)




# pred_labels=[2 for i in range(146)]
# pred_labels.extend([3 for i in range(188)])
# labels=[3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3,
#         3, 0, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3,
#         3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3,
#         3, 3,
#         2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3,
#         2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 3, 2, 2,
#         3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 2, 2, 2, 3, 2, 2, 0, 2, 0, 2,
#         3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 0, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 3,
#         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 0, 2, 2, 2, 3, 2, 2, 2,
#         2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 2, 2, 2, 2,
#         2, 2, 3, 3, 0, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
#         2, 3, 2, 2, 3, 3, 0, 2, 3, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2]

# cluster_eval(labels,pred_labels,2,4)
