from sklearn.metrics.cluster import pair_confusion_matrix,adjusted_rand_score,rand_score,normalized_mutual_info_score
import os
import numpy as np
from scipy.optimize import linear_sum_assignment



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
        
        
    return init_array, NMI_pred_labels, NMI_gt_labels, cluster_cocurrency_all_labels_array, cluster_dict, label_dict

            

def eval_single_file_result(task_name,file_name):
    known_class_dict={
        'agnews': 2,
        'yahoo': 5,
        'dbpedia': 7,
        'amazonproduct': 12
    }
    
    matrix=[]
    ture_label_count_dict={}
    pred_label_count_dict={}
    true_equal_pred_count_dict={}
        
    print("============={} Summary=============".format(file_name))
    with open(file_name,"r") as fr:
        for line in fr.readlines():
            line=line.split(",",4)[:4]
            matrix.append(line)
            #line[1] pred_label line[2] true_label
            pred_label=int(line[1])
            true_label=int(line[2])
            if pred_label not in pred_label_count_dict:
                pred_label_count_dict[pred_label] = 0
            if true_label not in ture_label_count_dict:
                ture_label_count_dict[true_label] = 0
                true_equal_pred_count_dict[true_label] = 0
            pred_label_count_dict[pred_label] += 1
            ture_label_count_dict[true_label] += 1
            if true_label == pred_label :
                true_equal_pred_count_dict[true_label] += 1
                
            
            
    np_m=np.array(matrix, dtype=int)

    all_true_labels = list(np_m[:,2])
    all_pred_labels = list(np_m[:,1])
    # ari f-score nmi
    (tn, fp), (fn, tp) =pair_confusion_matrix(all_true_labels, all_pred_labels)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    beta = 1
    f_score = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    all_nmi = normalized_mutual_info_score(all_true_labels, all_pred_labels)
    print("all ari     : ",round(ari,4)) 
    print("all f_score : ",round(f_score,4))
    print("all nmi     : ",round(all_nmi,4))
    print()
    
    known_class =known_class_dict[task_name]
    # pred_seen_class=
    total_p=0
    total_r=0
    seen_macro_f1 = 0
    for i in range(known_class):
        # precision
        tmp_p = true_equal_pred_count_dict[i] / pred_label_count_dict[i]
        # recall
        tmp_r = true_equal_pred_count_dict[i] / ture_label_count_dict[i]
        total_p += tmp_p
        total_r += tmp_r
        if tmp_p+tmp_r != 0:
            seen_macro_f1 += 2*tmp_p*tmp_r/(tmp_p+tmp_r)
    
    macro_seen_p = total_p / known_class
    macro_seen_r = total_r / known_class
    # macro_seen_f_score = (1 + beta**2) * (macro_seen_p * macro_seen_r / ((beta ** 2) * macro_seen_p + macro_seen_r))
    macro_seen_f_score = seen_macro_f1 / known_class
    print("seen macro precision : ", round(macro_seen_p,4))
    print("seen macro recall    : ", round(macro_seen_r,4))
    print("seen macro f-score   : ", round(macro_seen_f_score,4))
    print()
    
    unseen_gt_labels=[]
    unseen_pred_labels=[]
    for line in np_m:
        if line[1]>=known_class:
            unseen_gt_labels.append(line[2])
            unseen_pred_labels.append(line[1])
    
    
    preprocessed_matrix, NMI_pred_labels, NMI_gt_labels, cluster_cocurrency_all_labels_array, cluster_dict, label_dict = preprocess_for_hungarian_for_unlimited_clusters(unseen_gt_labels, unseen_pred_labels, known_class, known_class*2)
    row_ind,col_ind=linear_sum_assignment(preprocessed_matrix,True)
    rows_sum = np.sum(cluster_cocurrency_all_labels_array,axis=1)
    cluster_match_dict={} 
    h_total_p = 0
    h_total_r = 0
    unseen_macro_f1 = 0
    for i,row_id in enumerate(row_ind):
        pred_label = cluster_dict[row_id] 
        true_label = label_dict[col_ind[i]]
        cluster_match_dict[pred_label] = true_label
        # tmp_p
        hits = preprocessed_matrix[row_id,col_ind[i]] 
        tmp_p = hits / rows_sum[row_id]
        tmp_r = hits / ture_label_count_dict[true_label]
        h_total_p += tmp_p
        h_total_r += tmp_r
        if tmp_p+tmp_r != 0:
            unseen_macro_f1 += 2*tmp_p*tmp_r/(tmp_p+tmp_r)
        
        
    macro_unseen_p = h_total_p / known_class  
    macro_unseen_r = h_total_r / known_class
    macro_unseen_f_score = unseen_macro_f1/known_class
    print("eval1: unseen macro precision : ", round(macro_unseen_p,4))
    print("eval1: unseen macro recall    : ", round(macro_unseen_r,4))
    print("eval1: unseen macro f-score   : ", round(macro_unseen_f_score,4))  
    print()
    
        
    tp_unseen_gt_labels=[]
    tp_unseen_pred_labels=[]
    for line in np_m:
        if line[2]>=known_class: 
            tp_unseen_gt_labels.append(line[2])
            tp_unseen_pred_labels.append(line[1])
    (tn, fp), (fn, tp) =pair_confusion_matrix(tp_unseen_gt_labels, tp_unseen_pred_labels)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    beta = 1
    f_score = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    all_nmi = normalized_mutual_info_score(all_true_labels, all_pred_labels)
    print("eval2: unseen ari     : ", round(ari,4))
    print("eval2: unseen f_score : ", round(f_score,4))
    print("eval2: unseen nmi     : ", round(all_nmi,4))
    
    print()
    
if __name__ == "__main__" :
    eval_single_file_result("agnews","./cls_results/agnews/agnews_detach_224_1225_mt2_3layer_7600.csv")