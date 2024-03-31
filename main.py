# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:34:45 2021

@author: 99488
"""
import matplotlib.colors as mcolors
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import sys
sys.path.append(r'E:\OPT\research\fMRI\utils_for_all')
from common_utils import get_function_connectivity, fig_pcd_distrub, outliner_detect, keep_triangle_half
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import matplotlib 
import copy
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import seaborn as sns
# from direpack import sprm
# from pls_alex import PLSRegression, CCA
from sklearn.cross_decomposition import CCA,PLSRegression
from sklearn.metrics import r2_score
# import phik
from scipy import stats
from sklearn.decomposition import PCA,TruncatedSVD,NMF, SparsePCA, FactorAnalysis
from sklearn.manifold import MDS
from scipy.stats import spearmanr, pearsonr, ttest_ind
from scipy.stats import boxcox, yeojohnson
from scipy.stats import normaltest
import seaborn as sns
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
# from skbio.stats.distance import mantel
np.random.seed(9)

font = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 15}
# matplotlib.rc('font', **font)

def get_npi(data, name_list, no_NC=False):
    npi_list = []
    for name in name_list:
        if pd.api.types.is_float_dtype(data[name]):
            npi_sub_scale = data[name].fillna(0)
            if no_NC:
                npi_list.append(npi_sub_scale[data['DX']!='CN'].values)
            else:
                npi_list.append(npi_sub_scale.values)
        else:
            variable_names = set(data[name])
            t = 0
            for variable_name in variable_names:
                data[name][data[name] == variable_name] = t
                t = t + 1
            data[name] = data[name].astype('float')
            if no_NC:
                npi_list.append(data[name][data['DX']!='CN'].values)
            else:
                npi_list.append(data[name].values)
    return np.array(npi_list).T


def regress_out_confunder(data, confounder):

    beta = np.dot(np.dot(np.linalg.inv(np.dot(confounder.T, confounder)), confounder.T), data)
    feature_regout = np.zeros((data.shape[0], data.shape[1]))
    feature_regout = data - np.dot(confounder, beta)

    return feature_regout    
     
def CPM_reduce_dimension(feature, target):
    r_p_array = np.zeros((feature.shape[-1], 3))
    for i in range(feature.shape[-1]):
        r, p = pearsonr(feature[:,i], target)
        if  p < 0.05:
            r_p_array[i,0] = r 
            r_p_array[i,1] = p 
            r_p_array[i,2] = 1
    
    return r_p_array

def pca(data, components, corr_matrix):

    # features = X_scaled.T
    cov_matrix = corr_matrix
        # cov_matrix = np.corrcoef(features)
    
    values, vectors = np.linalg.eig(cov_matrix)
    
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
         
    projected = data.dot(vectors.T[:,:components])

    return explained_variances, vectors, projected

def MAD(data, percentile):
    M = np.median(data, 0)
    diff = abs(data - M)
    M2 = np.median(diff, 0)
    thre = np.percentile(M2, percentile)
    mask = M2>=thre 
    return mask, diff

def Ttest(group1, group2):
    t_p_array = np.zeros((group1.shape[-1],2))
    for i in range(group1.shape[-1]):
        t, p = ttest_ind(group1[:,i], group2[:,i])
        if p <= 0.05:
            t_p_array[i,0] = t
            t_p_array[i,1] = p
        
    return t_p_array[:,1] != 0

def get_net_net_connects(connections, net_info, net_nums = 7):
    
    nets = list(net_info.keys())
    new_conn = np.zeros((connections.shape[0], net_nums, connections.shape[-1]))
    # new_conn2 = np.zeros((connections.shape[0], net_nums, connections.shape[-1]))
    for i in range(len(nets)):
        new_conn[:,i,:] = new_conn[:,i,:] + np.mean(connections[:,net_info[nets[i]][0][0]:net_info[nets[i]][0][1],:],1)
        new_conn[:,i,:] = new_conn[:,i,:] + np.mean(connections[:,net_info[nets[i]][1][0]:net_info[nets[i]][1][1],:],1)
    # for i in range(len(nets)):
    #     new_conn2[i,:,:] = new_conn2[i,:,:] + np.mean(new_conn[net_info[nets[i]][0][0]:net_info[nets[i]][0][1],:,:],0)
    #     new_conn2[i,:,:] = new_conn2[i,:,:] + np.mean(new_conn[net_info[nets[i]][1][0]:net_info[nets[i]][1][1],:,:],0)

    return new_conn
#########fc pcd preprocess
subjects = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest.csv')['subjectID_Date']
mask = subjects.str.contains('d000', regex=False)
fc = sio.loadmat(r'E:\PHD\learning\research\AD\data\OASIS3\fc_NPI.mat')['interest_subjects_fc']
npi = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest.csv').iloc[:,[8,10,12,14,16,18,20,22,24,26]]
subjects = subjects[mask]
fc = fc[mask]
npi = npi[mask]

npi = npi.fillna(0)
net_info_shf = {'VIS': ([0,9],[50,58]), 'SMN': ([9,15],[58,66]), 'DAN': ([15,23],[66,73]), 'VAN': ([23,30],[73,78]), 
                                           'LIM': ([30,33],[78,80]), 'FPC': ([33,37],[80,89]), 'DMN':([37,50],[89,100])}
fc_2 = get_net_net_connects(fc, net_info_shf)
ca_components = 4
folds = 10
# fc_components_list = [10,20,30,40,50,60,70,80,90]
# fc_components_list = [30,50,70,100,150,200,250,300,350,400,450,500,550,600,650,700,750]
fc_components_list = [30,40,50,60,70,80,90,100,130,150,180,200,210,220,230,240,250]

# fc_components_list = [50]

# fc_components = 0
npi_components = 4
npi_reduce_method = 'pca'
fc_reduce_method2 = 'pca'
dimension_method = 'cca'
sparsity = True
normal_transf = False
subsample = False
subsample_ratio = 0/4
if sparsity:
    if dimension_method == 'cca':
        l1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # l1 = [0.8]
        # l1 = [0.9]
        # l1 = [0.1]

    elif dimension_method == 'pls':
        # l1 = [0.2,0.3,0.5,0.7,0.9]
        # l1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # l1 = [0.9]
        l1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

else:
    if dimension_method == 'cca':
        l1 = [0]
    elif dimension_method == 'pls':
        l1 = [0]

# half_feature_ = fc_2.reshape(fc_2.shape[0],fc_2.shape[1]*fc_2.shape[2])
half_feature_ = keep_triangle_half(fc.shape[1] * (fc.shape[1]-1)//2, fc.shape[0], fc)
npi = npi.values
# eng = matlab.engine.start_matlab()
# half_feature_ = np.random.random((838,4950))
# npi = np.random.random((838,10))

if subsample:
    non_zero_id = np.where(np.sum(npi,1)!=0)[0]
    zero_id = np.where(np.sum(npi,1)==0)[0]
    idx = np.r_[zero_id[1:int(len(non_zero_id)*subsample_ratio)], non_zero_id]
    np.random.shuffle(idx)
    npi = npi[idx,:]
    half_feature = half_feature_[idx,:]
else:
    half_feature = half_feature_
    subsample_ratio = 0
# sio.savemat(os.path.join(r'E:\PHD\learning\research\AD_two_modal', 'brain_fc.mat'), {'fc_feature': half_feature})
# sio.savemat(os.path.join(r'E:\PHD\learning\research\AD_two_modal', 'NPI.mat'), {'npi_feature': npi})
# kmeans = KMeans(n_clusters=2, random_state=0).fit(half_feature)
# labels = kmeans.labels_
# idex = np.where(labels == 1)[0]
# idex2 = np.where(labels == 0)[0]
# idex = np.r_[idex, idex2]
# kmeans.predict([[0, 0], [12, 3]])

# plt.figure(figsize =(15,15))
# ax = plt.gca()
# norm = mcolors.TwoSlopeNorm(vmin=half_feature.min(), vmax = half_feature.max(), vcenter=0)
# plt.imshow(half_feature[idex], cmap = 'RdBu_r', norm=norm)
# plt.colorbar()
# name = 'raw_fc'
# plt.title(name)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# scaler = StandardScaler()
# half_feature = scaler.fit_transform(half_feature.T).T
# plt.figure(figsize =(15,15))
# ax = plt.gca()
# norm = mcolors.TwoSlopeNorm(vmin=half_feature.min(), vmax = half_feature.max(), vcenter=0)
# plt.imshow(half_feature[idex], cmap = 'RdBu_r', norm=norm)
# plt.colorbar()
# name = 'raw_fc'
# plt.title(name)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
if normal_transf:
    for i in range(npi.shape[-1]):
        # npi[:,i] = np.sign(npi[:,i]-np.median(npi[:,i])) * np.power(abs(npi[:,i]-np.median(npi[:,i])), 1/3)
        npi[:,i] = yeojohnson(npi[:,i])[0]

if fc_reduce_method2 == 'none':
    fc_components_list = [4950]

# feature = regress_out_confunder(raw_feature, confounder)
for fc_components in fc_components_list:
    for c_ in l1:
        kf = KFold(n_splits=folds)
        r_tr_list = np.zeros((folds,ca_components))
        r_te_list = np.zeros((folds,ca_components))
        j=0
        trans_y_tr = []
        trans_y_te = []
        trans_x_tr = []
        trans_x_te = []
        weight_x = np.zeros((folds, fc_components, ca_components))
        # weight_x = np.zeros((folds, int(4950*(100-fc_components)/100), ca_components))
        weight_y = np.zeros((folds, npi_components, ca_components))    
        weight_npi = np.zeros((folds, npi.shape[-1], npi_components))    
        weight_fc = np.zeros((folds, half_feature_.shape[-1], fc_components)) 
            
        # te_save_path = r'E:\PHD\learning\research\AD_two_modal\result\new\roi_roi\python\{}\10_domains\subsample_zero{}_{}\l1{}\test_npi_all_method_{}_fmri_{}_CAcomp{}_fold{}'.format(
        #     dimension_method, subsample, subsample_ratio, c_, fc_reduce_method2, fc_components, ca_components, folds)        
        te_save_path = r'E:\PHD\learning\research\AD_two_modal\result\new\baseline\roi_roi\python\{}\10_domains\subsample_zero{}_{}\l1{}\test_method_norm_{}_npi{}_method_{}_fmri{}_CAcomp{}_fold{}'.format(
            dimension_method, subsample, subsample_ratio, c_, npi_reduce_method, npi_components, 
            fc_reduce_method2, fc_components, ca_components, folds)
        print(te_save_path)
        if c_ == 0.8 and fc_components == 50:
            a = 0
        if not os.path.exists(te_save_path):
            os.makedirs(te_save_path)
        else:
            continue
        # else:
        #     continue
        idx_tr, idx_te = [], []
        for train_index, test_index in kf.split(half_feature):
            X_train = half_feature[train_index]
            X_test = half_feature[test_index]
            y_train = npi[train_index]
            y_test = npi[test_index] 
            
            # scaler = MinMaxScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.transform(X_test)   
        
            # if subsample:
            if fc_reduce_method2 == 't_test':
                normal = half_feature_[zero_id[int(len(non_zero_id)*subsample_ratio):]]
                mask = Ttest(X_train, normal)
                # X_train = X_train[:,mask]
                # X_test = X_test[:,mask]         
                X_train = X_train[:,mask]
                X_test = X_test[:,mask]        
        
            elif fc_reduce_method2 == 'nmf_minmaxnorm':
                fmri_pca = NMF(n_components=fc_components)
                X_train = fmri_pca.fit_transform(abs(X_train))
                X_test = fmri_pca.transform(abs(X_test))
            elif fc_reduce_method2 == 'pca':
                fmri_pca = PCA(n_components=fc_components)
                X_train = fmri_pca.fit_transform(X_train)
                var = fmri_pca.explained_variance_ratio_
                X_test = fmri_pca.transform(X_test)
            elif fc_reduce_method2 == 'mad':
                percentile = fc_components
                mask, med = MAD(X_train, percentile)
                X_train = X_train[:,mask]       
                X_test = X_test[:,mask]
            elif fc_reduce_method2 == 'sparse_pca':
                fmri_pca = SparsePCA(n_components=fc_components, alpha=0.5)
                X_train = fmri_pca.fit_transform(X_train)
                var = fmri_pca.explained_variance_ratio_
                X_test = fmri_pca.transform(X_test)
            elif fc_reduce_method2 == 'fa':
                fmri_pca = FactorAnalysis(n_components=fc_components)
                X_train = fmri_pca.fit_transform(X_train)
                # var = fmri_pca.explained_variance_ratio_
                X_test = fmri_pca.transform(X_test)
            # else:
            #     fc_components = 0
        
            if npi_reduce_method == 'nmf':
                npi_reduce_model = NMF(n_components=npi_components, random_state=42)
                y_train = npi_reduce_model.fit_transform(y_train+1)
                y_test = npi_reduce_model.transform(y_test+1)
            
            elif npi_reduce_method == 'pca':
                npi_reduce_model = PCA(n_components=npi_components)
                y_train = npi_reduce_model.fit_transform(y_train)
                var = npi_reduce_model.explained_variance_ratio_
                y_test = npi_reduce_model.transform(y_test)
        
            elif npi_reduce_method == 'mds':
                npi_reduce_model = MDS(n_components=npi_components, metric = True)
                y_train = npi_reduce_model.fit_transform(y_train)
                y_test = npi_reduce_model.transform(y_test)
        
            elif npi_reduce_method == 'fa':
                npi_reduce_model = FactorAnalysis(n_components=npi_components)
                y_train = npi_reduce_model.fit_transform(y_train)
                # var = npi_reduce_model.explained_variance_ratio_
                y_test = npi_reduce_model.transform(y_test)
            elif npi_reduce_method == 'sparse_pca':
                npi_reduce_model = SparsePCA(n_components=npi_components, alpha=0.5)
                y_train = npi_reduce_model.fit_transform(y_train)
                # var = fmri_pca.explained_variance_ratio_
                y_test = npi_reduce_model.transform(y_test)
            elif npi_reduce_method == 'none':
                scaler = StandardScaler()
                y_train = scaler.fit_transform(y_train)
                y_test = scaler.transform(y_test)                 
            # weight_fc[j] = fmri_pca.components_.T
            # weight_npi[j] = npi_reduce_model.components_.T
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)             
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train)  
            y_test = scaler.transform(y_test)  
        
            if l1[0]:
                if dimension_method == 'cca':
                    spls = importr('PMA')
                    r = robjects.r
                    numpy2ri.activate()
                    out_cca_r = r['CCA'](X_train, y_train, typex="standard",typez="standard",K=ca_components,niter=1000,
                                 penaltyx=c_, penaltyz=1)#larger c, less sparsity
                    out = dict(zip(out_cca_r.names, map(list,list(out_cca_r))))
                    w_x = np.reshape(np.array(out['u']), (ca_components, X_train.shape[-1])).T
                    w_y = np.reshape(np.array(out['v']), (ca_components, y_train.shape[-1])).T
                    X_train_ = np.dot(X_train, w_x)
                    y_train_ = np.dot(y_train, w_y)   
                    X_test_ = np.dot(X_test, w_x)
                    y_test_ = np.dot(y_test, w_y)
                    x_load = np.dot(X_train_.T, X_train).T / np.diagonal(np.dot(X_train_.T, X_train_))
                    pctVar = sum(abs(x_load)*abs(x_load),1) / sum(sum(abs(X_train)*abs(X_train),1))
                    covmat = np.dot(np.dot(np.dot(w_x.T,X_train.T), y_train), w_y) #calculate covariance matrix
                    varE = np.diagonal(covmat) * np.diagonal(covmat) / sum(np.diagonal(covmat) * np.diagonal(covmat)) #calcualte covariance explained by each component
                    print(varE)

                elif dimension_method == 'pls':
                    importr('mixOmics')
                    r = robjects.r
                    numpy2ri.activate()
                    c = c_*X_train.shape[-1]
                    c_list = r['c'](c, c, c) 
                    out_pls_r = r['spls'](X_train, y_train, ncomp = ca_components, keepX = c_list)
                    # out_pls_r = r['pls'](X_train, y_train, ncomp = ca_components)
                    value = list(out_pls_r)
                    name = list(out_pls_r.names)
                    del name[14] #手动索引去掉一个null对象。。
                    del value[14]              
                    del name[0] #去掉没必要的为map省时间
                    del value[0]   
                    del name[0]
                    del value[0]       
                    # del name[10]
                    # del value[10]               
                    del name[-1]
                    del value[-1]         
                    del name[-1]
                    del value[-1]
                    value = list(map(list,value))
                    variate = list(map(list,value[5]))
                    X_train_ = np.reshape(np.array(variate[0]), (ca_components, X_train.shape[0])).T
                    y_train_ = np.reshape(np.array(variate[1]), (ca_components, X_train.shape[0])).T
                    loading = list(map(list,value[6]))
                    w_x = np.reshape(np.array(loading[0]), (ca_components, X_train.shape[1])).T
                    w_y = np.reshape(np.array(loading[1]), (ca_components, y_train.shape[1])).T
                    X_test_ = np.dot(X_test, w_x)
                    y_test_ = np.dot(y_test, w_y)
                    x_load = np.dot(X_train_.T, X_train).T / np.diagonal(np.dot(X_train_.T, X_train_))
                    pctVar = sum(abs(x_load)*abs(x_load),1) / sum(sum(abs(X_train)*abs(X_train),1))
                    # pctVar = sum(abs(w_x)*abs(w_x),1) / sum(sum(abs(X_train_)*abs(X_train_),1))
                              # sum(abs(Yloadings).^2,1) ./ sum(sum(abs(Y0).^2,1))]
                    variance = list(map(list,value[-1]))
                    covmat = np.dot(np.dot(np.dot(w_x.T,X_train.T), y_train), w_y) #calculate covariance matrix
                    varE = np.diagonal(covmat) * np.diagonal(covmat) / sum(np.diagonal(covmat) * np.diagonal(covmat)) #calcualte covariance explained by each component
                    print(varE)
            else:
                if dimension_method == 'cca':
                    pls = CCA(n_components=ca_components)
                    X_train_, y_train_ = pls.fit_transform(X_train, y_train) 
                    X_test_, y_test_ = pls.transform(X_test, y_test) 
                    # pls.fit(X_train, y_train, cu = 0, cv=0) 
        
                elif dimension_method == 'pls':
                    pls = PLSRegression(n_components=ca_components)
                    X_train_, y_train_ = pls.fit_transform(X_train, y_train) 
                    X_test_, y_test_ = pls.transform(X_test, y_test) 
                w_x = pls.x_weights_
                w_y = pls.y_weights_
            trans_y_tr.append(y_train_)
            trans_y_te.append(y_test_)
            trans_x_tr.append(X_train_)
            trans_x_te.append(X_test_)
            idx_tr.append(train_index) 
            idx_te.append(test_index)
          
            # weight_x[j,:w_x.shape[0]] = w_x
            weight_x[j] = w_x
            weight_y[j] = w_y
            for i in range(ca_components):
                r_train = pearsonr(X_train_[:,i], y_train_[:,i])
                r_test = pearsonr(X_test_[:,i], y_test_[:,i])  
                r_tr_list[j,i] = r_train[0]
                r_te_list[j,i] = r_test[0]
            j = j+1
        
        print(trans_x_te[0].shape)
        print(trans_y_te[0].shape)
        x_te_final = np.concatenate((trans_x_te[0], trans_x_te[1], trans_x_te[2], trans_x_te[3], trans_x_te[4], 
                                     trans_x_te[5], trans_x_te[6], trans_x_te[7], trans_x_te[8], trans_x_te[9]),0)
        x_tr_final = np.concatenate((trans_x_tr[0], trans_x_tr[1], trans_x_tr[2], trans_x_tr[3], trans_x_tr[4],
                                     trans_x_tr[5], trans_x_tr[6], trans_x_tr[7], trans_x_tr[8], trans_x_tr[9]),0)
        y_te_final = np.concatenate((trans_y_te[0], trans_y_te[1], trans_y_te[2], trans_y_te[3], trans_y_te[4],
                                     trans_y_te[5], trans_y_te[6], trans_y_te[7], trans_y_te[8], trans_y_te[9]),0)
        y_tr_final = np.concatenate((trans_y_tr[0], trans_y_tr[1], trans_y_tr[2], trans_y_tr[3], trans_y_tr[4],
                                     trans_y_tr[5], trans_y_tr[6], trans_y_tr[7], trans_y_tr[8], trans_y_tr[9]),0)
        r_total_tr = list(pearsonr(x_tr_final[:,i], y_tr_final[:,i])[0] for i in range(ca_components))
        r_total_te = list(pearsonr(x_te_final[:,i], y_te_final[:,i])[0] for i in range(ca_components))
        print(varE)
        print(r_total_tr)
        r_save_file = os.path.join(te_save_path, 'r.txt')
        for n, r in enumerate(r_total_tr):
            f = open(r_save_file,'a')
            f.write('train comp {} r: {}\n'.format(n,r))
            f.close()   
        for n, r in enumerate(r_total_te):
            f = open(r_save_file,'a')
            f.write('test comp {} r: {}\n'.format(n,r))
            f.close()
            
        print(r_total_te)
        sio.savemat(os.path.join(te_save_path, 'output.mat'), {'trans_x_te': trans_x_te, 'trans_x_tr': trans_x_tr, 
                                                               'trans_y_te': trans_y_te, 'trans_y_tr': trans_y_tr,
                                                               'pls_x_weight': weight_x, 'pls_y_weight': weight_y,
                                                               'variance': varE,
                                                               'weight_npi': weight_npi, 
                                                               # 'weight_fc': weight_fc,
                                                               'idx_tr': idx_tr, 'idx_te': idx_te})
        plt.figure(figsize =(15,15))
        axes = plt.gca()
        axes.boxplot(np.array(r_tr_list),patch_artist=True) #描点上色
        plt.savefig(os.path.join(te_save_path, 'train_perfomance.png'))
        plt.show() 
    
        plt.figure(figsize =(15,15))
        axes = plt.gca()
        axes.boxplot(np.array(r_te_list),patch_artist=True) #描点上色
        # plt.text(2.999, 0.1, 'comp1: {:.3}\ncomp2: {:.3}\ncomp3: {:.3}\ncomp4: {:.3}'.format(
        #     r_total_te[0], r_total_te[1], r_total_te[2], r_total_te[3]))  
        plt.text(2.999, 0.1, 'comp1: {:.3}\ncomp2: {:.3}\ncomp3: {:.3}'.format(
                    r_total_te[0], r_total_te[1], r_total_te[2]))
        plt.savefig(os.path.join(te_save_path, 'test_perfomance.png'))
        plt.show() 
    
        plt.figure(figsize =(10,10))
        sns.kdeplot(npi[:,0], shade=True)
        sns.rugplot(npi[:,0])
        name = 'npi_component1'
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(10,10))
        sns.kdeplot(npi[:,1], shade=True)
        sns.rugplot(npi[:,1])
        name = 'npi_component2'
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        if npi_reduce_method != 'mds' and npi_reduce_method != 'none':
            plt.figure(figsize =(15,15))
            ax = plt.gca()
            plt.imshow(npi_reduce_model.components_, cmap = 'RdBu')
            plt.colorbar()
            name = 'npi_{}_loadings'.format(npi_reduce_method)
            plt.title(name)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                      rotation_mode="anchor")
            plt.xticks(np.arange(10), ['Delusions', 'Hallucinations', 'Agitation', 'Depression', 'Anxiety', 'Euphoria', 
                         'Apathy', 'Disinhibition', 'Irritability', 'Aberrant motor\nbehavior'])
            plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(half_feature[:,0], shade=True)
        # sns.rugplot(half_feature[:,0])
        # name = 'fc_component1'
        # plt.title(name)
        # plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(half_feature[:,1], shade=True)
        # sns.rugplot(half_feature[:,1])
        # name = 'fc_component2'
        # plt.title(name)
        # plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        # plt.figure(figsize =(15,15))
        # ax = plt.gca()
        # norm = mcolors.TwoSlopeNorm(vmin=half_feature.min()-0.1, vmax = half_feature.max(), vcenter=0)
        # plt.imshow(half_feature, cmap = 'RdBu', norm=norm)
        # plt.colorbar()
        # name = 'fc_after{}mapping'.format(method)
        # plt.title(name)
        # plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=X_train[:200,:].min()-0.1, vmax = X_train[:200,:].max(), vcenter=0)
        plt.imshow(X_train[:200,:], cmap = 'RdBu', norm=norm)
        plt.colorbar()
        name = 'fc_after{}mapping_first200subjects'.format(npi_reduce_method)
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        # plt.figure(figsize =(15,15))
        # ax = plt.gca()
        # norm = mcolors.TwoSlopeNorm(vmin=fmri_pca.components_[:,-200:].min()-0.1, vmax = fmri_pca.components_[:,-200:].max(), vcenter=0)
        # plt.imshow(fmri_pca.components_[:,-200:], cmap = 'RdBu', norm=norm)
        # plt.colorbar()
        # name = 'fc_weights_last200fc'
        # plt.title(name)
        # plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(25,15))
        ax = plt.gca()
        # norm = mcolors.TwoSlopeNorm(vmin=w_x.min(), vmax = w_x.max(), vcenter=0)
        # plt.imshow(w_x[:100,:], cmap = 'RdBu', norm=norm)
        plt.imshow(w_x[:100,:], cmap = 'RdBu')
        plt.colorbar()
        name = 'cca_xloadings'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        try:
            norm = mcolors.TwoSlopeNorm(vmin=w_y.min(), vmax = w_y.max(), vcenter=0)
            plt.imshow(w_y, cmap = 'RdBu', norm=norm)
        except ValueError:
            plt.imshow(w_y, cmap = 'RdBu')
        plt.colorbar()
        name = 'cca_yloadings'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=npi.min()-0.1, vmax = npi.max(), vcenter=0)
        plt.imshow(npi, cmap = 'RdBu', norm=norm)
        plt.colorbar()
        name = 'npi_after{}mapping'.format(npi_reduce_method)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=npi[:200].min()-0.1, vmax = npi[:200].max(), vcenter=0)
        plt.imshow(npi[:200], cmap = 'RdBu', norm=norm)
        plt.colorbar()
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        name = 'npi_after{}mapping_first200'.format(npi_reduce_method)
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        # plt.figure(figsize =(15,15))
        # ax = plt.gca()
        # norm = mcolors.TwoSlopeNorm(vmin=half_feature[:,:1000].min(), vmax = half_feature[:,:1000].max(), vcenter=0)
        # plt.imshow(half_feature[:,:1000], cmap = 'RdBu', norm=norm)
        # plt.colorbar()
        # name = 'raw_first1000_fc'
        # plt.title(name)
        # plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=X_train_[:100,:].min(), vmax = X_train_[:100,:].max(), vcenter=0)
        plt.imshow(X_train_[:100,:], cmap = 'RdBu', norm=norm)
        plt.colorbar()
        name = 'cca_X_train_transform_first100'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=y_train_[:100,:].min(), vmax = y_train_[:100,:].max(), vcenter=0)
        plt.imshow(y_train_[:100,:], cmap = 'RdBu', norm=norm)
        plt.colorbar()
        name = 'cca_y_train_transform_first100'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        norm = mcolors.TwoSlopeNorm(vmin=X_test_.min(), vmax = X_test_.max(), vcenter=0)
        plt.imshow(X_test_, cmap = 'RdBu', norm=norm)
        plt.colorbar()
        name = 'cca_x_test_transform'
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        # norm = mcolors.TwoSlopeNorm(vmin=y_test_.min(), vmax = y_test_.max(), vcenter=0)
        plt.imshow(y_test_, cmap = 'RdBu')
        plt.colorbar()
        name = 'cca_y_test_transform'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))    
           
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        plt.scatter(X_train_[:,0], y_train_[:,0])
        name = 'tr_cca_comp1_xy'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name))) 
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        plt.scatter(x_te_final[:,0], y_te_final[:,0])
        name = 'te_cca_comp1_xy'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name))) 
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        plt.scatter(X_train_[:,1], y_train_[:,1])
        name = 'tr_cca_comp2_xy'
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))    
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        name = 'te_cca_comp2_xy'
        plt.scatter(x_te_final[:,1], y_te_final[:,1])
        plt.title(name)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(os.path.join(te_save_path, '{}.png'.format(name)))
        
        plt.close('all')
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(npi[:,1], shade=True)
        # sns.rugplot(npi[:,1])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(npi[:,2], shade=True)
        # sns.rugplot(npi[:,2])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(npi[:,3], shade=True)
        # sns.rugplot(npi[:,3])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_tr[0,:,0], shade=True)
        # sns.rugplot(trans_y_tr[0,:,0])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_tr[0,:,1], shade=True)
        # sns.rugplot(trans_y_tr[0,:,1])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_tr[0,:,2], shade=True)
        # sns.rugplot(trans_y_tr[0,:,2])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_tr[0,:,3], shade=True)
        # sns.rugplot(trans_y_tr[0,:,3])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_te[0,:,0], shade=True)
        # sns.rugplot(trans_y_te[0,:,0])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_te[0,:,1], shade=True)
        # sns.rugplot(trans_y_te[0,:,1])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_te[0,:,2], shade=True)
        # sns.rugplot(trans_y_te[0,:,2])
        
        # plt.figure(figsize =(10,10))
        # sns.kdeplot(trans_y_te[0,:,3], shade=True)
        # sns.rugplot(trans_y_te[0,:,3])
        
