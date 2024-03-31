# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:41:18 2021

@author: 99488
"""

import matplotlib.colors as mcolors
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import sys
sys.path.append(r'F:\OPT\research\fMRI\utils_for_all')
from common_utils import get_function_connectivity, fig_pcd_distrub, outliner_detect, keep_triangle_half,\
    vector_to_matrix, get_rois_label, get_net_net_connect, heatmap, setup_seed, t_test, get_fc, read_singal_fmri_ts
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import matplotlib 
import copy
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.cross_decomposition import CCA,PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import phik
from scipy import stats
from sklearn.decomposition import PCA,TruncatedSVD,NMF, SparsePCA,FactorAnalysis, SparsePCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr, ttest_ind
from statsmodels.stats import multitest
from scipy.stats import boxcox, yeojohnson, normaltest
import seaborn as sns
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
# from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy.stats import levene
from statsmodels.robust.scale import mad
# from skimage.metrics import structural_similarity as ssim
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from bids.layout import parse_file_entities

np.random.seed(6)
font = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 15}
font2 = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 35}
# font2 = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 10}
font3 = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font2)

def MAD(data, percentile):
    M2 = mad(data, axis=0)
    thre = np.percentile(M2, percentile)
    mask = M2>=thre 
    return mask

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

def Ttest(group1, group2):
    t_p_array = np.zeros((group1.shape[-1],2))
    for i in range(group1.shape[-1]):
        t, p = ttest_ind(group1[:,i], group2[:,i])
        if p <= 0.05:
            t_p_array[i,0] = t
            t_p_array[i,1] = p
        
    return t_p_array[:,1] != 0

def get_roi_net_connects(connections, net_info, net_nums = 7):
    
    nets = list(net_info.keys())
    new_conn = np.zeros((connections.shape[0], net_nums, connections.shape[-1]))
    for i in range(len(nets)):
        new_conn[:,i,:] = new_conn[:,i,:] + np.mean(connections[:,net_info[nets[i]][0][0]:net_info[nets[i]][0][1],:],1)
        new_conn[:,i,:] = new_conn[:,i,:] + np.mean(connections[:,net_info[nets[i]][1][0]:net_info[nets[i]][1][1],:],1)


    return new_conn

def concate(matrix):
    if len(matrix.shape) == 2:
        for i, array in enumerate(matrix):
            if i == 0:
                out = array[0]
            else:
                out = np.concatenate((out, array[0])) 
        return out
    else:
        out_ = []
        for i, array in enumerate(matrix):
            out = []
            for j in range(matrix.shape[-1]):
                out.extend(list(array[:,j]))
            out_.append(out)
        return np.array(out_).T
    
def get_variance_explained(model, data):
    fa_loadings = model.components_.T    # loadings

    # variance explained
    total_var = data.var(axis=0).sum()  # total variance of original variables,
                                        # equal to no. of vars if they are standardized
    var_exp = np.sum(fa_loadings**2, axis=0)
    prop_var_exp = var_exp/total_var
    return var_exp, prop_var_exp

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
    
oas_pcd_info = sio.loadmat(r'E:\PHD\learning\research\AD\data\OASIS3\fMRIdata_OASIS3_struct.mat')
tvar = oas_pcd_info['OASIS3_Phenotypic_struct']

PCD_data = []
pcd_keys = ['subjectID_Date', 'dx1', 'Age', 'apoe', 'NPIQINF', 'NPIQINFX', 'DEL',
             'DELSEV', 'HALL', 'HALLSEV', 'AGIT', 'AGITSEV', 'DEPD', 'DEPDSEV', 'ANX', 
             'ANXSEV', 'ELAT', 'ELATSEV', 'APA', 'APASEV', 'DISN', 'DISNSEV', 'IRR', 
             'IRRSEV', 'MOT', 'MOTSEV', 'NITE', 'NITESEV', 'APP', 'APPSEV', 'mmse',
             'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
             'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
             'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON']
for idx in range(len(tvar)):
    sub_data = []
    for name in pcd_keys:
        value = tvar[idx][0][name]
        for i in range(3):
            if isinstance(value, np.ndarray):
                try:
                    value = value[0]
                except IndexError:
                    if name == 'DX':
                        value = 'nan'
                    else:
                        value = "error"
            else:
                if name == 'subjectID_Date' or name == 'dx1':
                    value = str(value)
                sub_data.append(value)
                break
    PCD_data.append(sub_data)
PCD_data = np.array(PCD_data)
pcd_info= pd.Series(PCD_data[:,0]).str.split('d', expand = True)
pcd_day = pcd_info.iloc[:,1].str.split('d', expand = True).iloc[:,-1].astype(int).values
pcd_sub = pcd_info.iloc[:,0]
pcd_sub = np.array([info[:8] for info in pcd_sub])

info = sio.loadmat(r'E:\PHD\learning\research\AD\data\OASIS3\Updated_fc_240328.mat')
sess_unique = info['sess']
fc_unique = info['fc']
sub_unique_ = info['sub']
all_sess = info['all_sess']
all_day = np.array([day[0] for day in info['all_day'].squeeze()])
output = sio.loadmat(r'E:\PHD\learning\research\AD_two_modal\result\multi_run\baseline\roi_100\python\cca\12_domains_OAS_norm_True\subsample_zeroTrue_0.0\l10.6\test_method_norm_pca_npi7_method_none_fmri4950_CAcomp4_fold10\output.mat')

trans_x_te_ = output['trans_x_te'].T
trans_x_tr = output['trans_x_tr'].T
trans_y_te_ = output['trans_y_te'].T
trans_y_tr = output['trans_y_tr'].T

trans_x_te = np.zeros((177, 4))
trans_y_te = np.zeros((177, 4))

for i in range(output['idx_te'].shape[1]):
    idx_tr = output['idx_tr'][:,i][0].squeeze()
    idx_te = output['idx_te'][:,i][0].squeeze()
    trans_x_te[idx_te,:] = trans_x_te_[i,:][0]
    trans_y_te[idx_te,:] = trans_y_te_[i,:][0]

subjects_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest_multirun_Schaefer200_moreITEM.csv')['subjectID_Date']
subjects_oas_array = subjects_oas.values
sub_names = np.array([i.split('_')[0] for i in subjects_oas_array])
sub_names_uniq = np.array(list(set(list(sub_names))))
sub_names_uniq.sort()
_, sub_names_uniq_id, sub_names_id = np.intersect1d(sub_names_uniq, sub_names, return_indices=True)

mask = subjects_oas.str.contains('d000', regex=False)
fc_oas = sio.loadmat(r'E:\PHD\learning\research\AD\data\OASIS3\fc_NPI_roi_multirun_zscore.mat')['interest_subjects_fc']
npi_oas_raw = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest_multirun_Schaefer200_moreITEM2.csv').iloc[:,[8,10,12,14,16,18,20,22,24,26,28,30]]
pup_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\PUP.csv')
pup_oas_subject = pup_oas['PUP_PUPTIMECOURSEDATA ID'].values
pup_oas_BP_SUVR = pup_oas[['Centil_fBP_TOT_CORTMEAN', 'Centil_fSUVR_TOT_CORTMEAN', 'Centil_fBP_rsf_TOT_CORTMEAN', 'Centil_fSUVR_rsf_TOT_CORTMEAN']].values
health_hist_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\UDS-A5-HealthHistory.csv')
health_hist_oas_subject = health_hist_oas['UDS_A5SUBHSTDATA ID'].values
health_hist_oas_info = health_hist_oas[['CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR',
                                        'PD', 'PDOTHR',	'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES',
                                        'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100', 'ABUSOTHR', 
                                        'PSYCDIS']].values #PACKSPER	QUITSMOK	ABUSOTHR	ABUSX	PSYCDIS
cvd_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\UDS-B2-HIS_CVD.csv')
cvd_oas_subject = cvd_oas['UDS_B2HACHDATA ID'].values
cvd_oas_info = cvd_oas[['ABRUPT', 'STEPWISE', 'SOMATIC', 'EMOT', 'HXHYPER', 'HXSTROKE', 'FOCLSYM', 'FOCLSIGN', 'HACHIN', 'CVDCOG', 'STROKCOG', 'CVDIMAG']].values 
updrs_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\UDS-B3-UPDRS.csv')
updrs_oas_subject = updrs_oas['UDS_B3UPDRSDATA ID'].values
updrs_oas_info = updrs_oas[['PDNORMAL', 'SPEECH', 'FACEXP', 'TRESTFAC', 'TRESTRHD', 'TRESTLHD', 'TRESTRFT', 'TRESTLFT', 'TRACTRHD', 'TRACTLHD',
                            'RIGDNECK', 'RIGDUPRT', 'RIGDUPLF', 'RIGDLORT', 'RIGDLOLF', 'TAPSRT', 'TAPSLF', 'HANDMOVR', 'HANDMOVL', 'HANDALTR',
                            'HANDALTL', 'LEGRT', 'LEGLF', 'ARISING', 'POSTURE', 'GAIT', 'POSSTAB', 'BRADYKIN']].values 
gds_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\UDS-B6-GDS.csv')
gds_oas_subject = gds_oas['UDS_B6BEVGDSDATA ID'].values
gds_oas_info = gds_oas[['GDS']].values 
faq_oas = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\UDS-B7-FAQ.csv')
faq_oas_subject = faq_oas['UDS_B7FAQDATA ID'].values
faq_oas_info = faq_oas[['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']].values 

subjects_oas_bl = subjects_oas[mask]
npi_oas_raw = npi_oas_raw.fillna(0)
fc_oas_bl = fc_oas[mask]
npi_oas_bl = npi_oas_raw[mask]

fc_adni = sio.loadmat(r'E:\PHD\learning\research\AD\data\ADNI\fc.mat')['interest_subjects_fc']
npi_adni = pd.read_csv(r'E:\PHD\learning\research\AD\data\ADNI\pcd_interest.csv').iloc[:,[29,31,33,35,37,39,41,43,45,47]]
subjects_adni = pd.read_csv(r'E:\PHD\learning\research\AD\data\ADNI\pcd_interest.csv')['TimePoint']
mask2 = subjects_adni.str.contains('bl', regex=False)
subjects_adni = subjects_adni[mask2]
npi_adni = npi_adni.fillna(0)
fc_adni = fc_adni[mask2]
npi_adni = npi_adni[mask2]

fc = fc_oas_bl
npi = npi_oas_bl.values
# fc = np.r_[fc_oas, fc_adni]
# npi = np.r_[npi_oas_bl.values, npi_adni.values]
net_info_shf = {'VIS': ([0,9],[50,58]), 'SMN': ([9,15],[58,66]), 'DAN': ([15,23],[66,73]), 'VAN': ([23,30],[73,78]), 
                                           'LIM': ([30,33],[78,80]), 'FPC': ([33,37],[80,89]), 'DMN':([37,50],[89,100])}

continue_pcd = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest_multirun_Schaefer200_moreITEM.csv')[['EDU','mmse', 'DIGIF', 'DIGIB',
                                                                                    'ANIMALS', 'VEG', 'TRAILA', 'TRAILB', 
                                                                                    'WAIS', 'LOGIMEM', 'MEMUNITS', 'BOSTON', 'sumbox']]
decret_pcd = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest_multirun_Schaefer200_moreITEM2.csv')[['cdr', 'SEX', 'Age',
                                                                                                                      'commun', 'homehobb',
                                                                                                                      'judgment', 'memory', 'orient', 'perscare', 'apoe']]
label = pd.read_csv(r'E:\PHD\learning\research\AD\data\OASIS3\pcd_interest_multirun_Schaefer200_moreITEM.csv')['dx1']


continue_pcd_bl = continue_pcd[mask]
decret_pcd_bl = decret_pcd[mask]
label_bl = label[mask]

decret_pcd_bl = decret_pcd_bl.values
# null_mask = np.ones((len(decret_pcd_bl)))==1#np.sum(np.isnan(decret_pcd_bl),-1)!=1
null_mask = np.sum(np.isnan(decret_pcd_bl[:,:2]),-1)!=1
fc = fc[null_mask,:,:]
npi = npi[null_mask,:,]
decret_pcd_bl = decret_pcd_bl[null_mask,:]
continue_pcd_bl = continue_pcd_bl.values[null_mask,:]
label_bl = label_bl.values[null_mask]
subjects_oas_bl = subjects_oas_bl[null_mask]

comp = 1
vis = 'fdr'
vari2 = 'brain'
roi_num = 100
name_ = 'cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10'
te_save_path = r'E:\PHD\learning\research\AD_two_modal\result\revised'
# E:\PHD\learning\research\AD_two_modal\result\multi_run\baseline\roi_100\python\cca\12_domains_OAS_norm_True\subsample_zeroTrue_0.0\l10.6\test_method_norm_pca_npi7_method_none_fmri4950_CAcomp4_fold10
ca_components = 7
folds = 10
fc_components = 4950
npi_components = 7
npi_reduce_method = 'pca'
fc_reduce_method2 = 'none'
if fc_reduce_method2 == 'none':
    decomp = False
else:
    decomp = True

dimension_method = 'cca'
sparsity = True
subsample = True
subsample_ratio = 0/4
feature_norm = True

if sparsity:
    if dimension_method == 'cca':
        c_ = 0.6
    elif dimension_method == 'pls':
        c_ = 0.6
else:
    if dimension_method == 'cca':
        l1 = [0]
    elif dimension_method == 'pls':
        l1 = [0]
half_feature = keep_triangle_half(fc.shape[1] * (fc.shape[1]-1)//2, fc.shape[0], fc)
npi_raw = npi
npi_oas_raw = npi_oas_raw.values
continue_pcd = continue_pcd.values
decret_pcd = decret_pcd.values
label = label.values
subjects_oas = subjects_oas.values
half_feature_oas = keep_triangle_half(fc_unique.shape[1] * (fc_unique.shape[1]-1)//2, fc_unique.shape[0], fc_unique)

if subsample:
    non_zero_id = np.where(np.sum(npi_raw,1)!=0)[0]
    zero_id = np.where(np.sum(npi_raw,1)==0)[0]
    idx = np.r_[zero_id[:int(len(non_zero_id)*subsample_ratio)], non_zero_id]
    np.random.shuffle(idx)
    npi = npi_raw[idx,:]
    npi_raw2 = npi_raw[idx,:]
    continue_pcd_bl = continue_pcd_bl[idx,:]
    decret_pcd_bl = decret_pcd_bl[idx,:]
    half_feature = half_feature[idx,:]
    label_bl = label_bl[idx]
    pcd = decret_pcd_bl
    subjects_oas_bl = subjects_oas_bl.iloc[idx]
    subjectsbl = [sub[:8] for sub in subjects_oas_bl]
    
    sub_f, _, _ = np.intersect1d(subjectsbl, sub_unique_, return_indices=True)
    sub_more = []
    sess_more = []
    days_more = []
    fc_more = []
    dx_more = []
    for sub in sub_f:
        mask = (sub_unique_ == sub) & (sess_unique != 'ses-M000')
        sub_more.extend(list(sub_unique_[mask]))
        sess_more.extend(list(sess_unique[mask]))
        pcd_info= pd.Series(PCD_data[:,0]).str.split('d', expand = True)
        pcd_day = pcd_info.iloc[:,1].str.split('d', expand = True).iloc[:,-1].astype(int).values
        pcd_sub = pcd_info.iloc[:,0]
        pcd_sub = np.array([info[:8] for info in pcd_sub])
        for sess in sess_unique[mask]:
            day = int(all_day[all_sess == sess][0][1:])
            diff = abs(pcd_day - day)
            days_more.append(day)
            dx = PCD_data[(pcd_sub==sub)&(diff <= 600),1][0]
            dx_more.append(dx)
        fc_more.extend(list(half_feature_oas[mask]))
    sub_more = np.array(sub_more)
    sess_more = np.array(sess_more)
    fc_more = np.array(fc_more)
    days_more = np.array(days_more)
    dx_more = np.array(dx_more)

else:
    npi = npi_raw
    npi_raw2 = npi_raw
    continue_pcd_bl = continue_pcd_bl
    decret_pcd_bl = decret_pcd_bl
    half_feature = half_feature
    label_bl = label_bl
    pcd = decret_pcd_bl

pup_oas_BP_SUVR_list = np.zeros((len(subjects_oas_bl), 4))
heal_his_list = np.zeros((len(subjects_oas_bl), health_hist_oas_info.shape[-1]))
cvd_list = np.zeros((len(subjects_oas_bl), cvd_oas_info.shape[-1]))
updrs_list = np.zeros((len(subjects_oas_bl), updrs_oas_info.shape[-1]))
gds_list = np.zeros((len(subjects_oas_bl), gds_oas_info.shape[-1]))
faq_list = np.zeros((len(subjects_oas_bl), faq_oas_info.shape[-1]))
for j, subject in enumerate(subjects_oas_bl):
    sub_fc, day_fc = subject.split('_')
    for i, pup_sub in enumerate(pup_oas_subject):
        pup_sub_info = pup_sub.split('_')
        sub_pup, day_pup = pup_sub_info[0], int(pup_sub_info[-1].split('d')[-1])
        if sub_pup == sub_fc:
            if day_pup<100:
                pup_oas_BP_SUVR_list[j,:] = pup_oas_BP_SUVR[i,:]
            else:
                pup_oas_BP_SUVR_list[j,:] = np.nan
            break
        else:
            pup_oas_BP_SUVR_list[j,:] = np.nan
    for i, his_sub in enumerate(health_hist_oas_subject):
        his_sub_info = his_sub.split('_')
        his_pup, day_his = his_sub_info[0], int(his_sub_info[-1].split('d')[-1])
        if his_pup == sub_fc:
            if day_his<100:
                heal_his_list[j,:] = health_hist_oas_info[i,:]
            else:
                heal_his_list[j,:] = np.nan
            break
        else:
            heal_his_list[j,:] = np.nan
    for i, his_sub in enumerate(cvd_oas_subject):
        his_sub_info = his_sub.split('_')
        his_pup, day_his = his_sub_info[0], int(his_sub_info[-1].split('d')[-1])
        if his_pup == sub_fc:
            if day_his<100:
                cvd_list[j,:] = cvd_oas_info[i,:]
            else:
                cvd_list[j,:] = np.nan
            break
        else:
            cvd_list[j,:] = np.nan
    for i, his_sub in enumerate(updrs_oas_subject):
        his_sub_info = his_sub.split('_')
        his_pup, day_his = his_sub_info[0], int(his_sub_info[-1].split('d')[-1])
        if his_pup == sub_fc:
            if day_his<100:
                updrs_list[j,:] = updrs_oas_info[i,:]
            else:
                updrs_list[j,:] = np.nan
            break
        else:
            updrs_list[j,:] = np.nan
    for i, his_sub in enumerate(gds_oas_subject):
        his_sub_info = his_sub.split('_')
        his_pup, day_his = his_sub_info[0], int(his_sub_info[-1].split('d')[-1])
        if his_pup == sub_fc:
            if day_his<100:
                gds_list[j,:] = gds_oas_info[i,:]
            else:
                gds_list[j,:] = np.nan
            break
        else:
            gds_list[j,:] = np.nan
    for i, his_sub in enumerate(faq_oas_subject):
        his_sub_info = his_sub.split('_')
        his_pup, day_his = his_sub_info[0], int(his_sub_info[-1].split('d')[-1])
        if his_pup == sub_fc:
            if day_his<100:
                faq_list[j,:] = faq_oas_info[i,:]
            else:
                faq_list[j,:] = np.nan
            break
        else:
            faq_list[j,:] = np.nan       
heal_his_list[heal_his_list==9] = np.nan
cvd_list[cvd_list==8] = np.nan
updrs_list[updrs_list==8] = np.nan
faq_list[(faq_list==8) | (faq_list==9)] = np.nan
continue_pcd_bl = np.c_[decret_pcd_bl[:,2],continue_pcd_bl, pup_oas_BP_SUVR_list, gds_list]
decret_pcd_bl = np.c_[decret_pcd_bl, heal_his_list, cvd_list, updrs_list, faq_list]

if feature_norm:
    scaler = StandardScaler()
    half_feature = scaler.fit_transform(half_feature.T).T
    half_feature_oas_more = scaler.fit_transform(fc_more.T).T

###################used for visualization
if fc_reduce_method2 == 'pca':
    fmri_pca = PCA(n_components=fc_components)
    X_train = fmri_pca.fit_transform(half_feature)
    var = fmri_pca.explained_variance_ratio_
elif fc_reduce_method2 == 'fa':
    fmri_pca = FactorAnalysis(n_components=fc_components)
    X_train = fmri_pca.fit_transform(half_feature)
elif fc_reduce_method2 == 'mad':
    percentile = fc_components
    mask = MAD(half_feature, percentile)
    X_train = half_feature[:,mask]       
else:
    X_train = half_feature
    
if npi_reduce_method == 'pca':
    npi_reduce_model = PCA(n_components=npi_components)
    npi = npi_reduce_model.fit_transform(npi)
    # npi_more = npi_reduce_model.transform(npi_more)
elif npi_reduce_method == 'nmf':
    npi_reduce_model = NMF(n_components=npi_components, random_state=42)
    npi = npi_reduce_model.fit_transform(npi+1)
elif npi_reduce_method == 'fa':
    npi_reduce_model = FactorAnalysis(n_components=npi_components)
    npi = npi_reduce_model.fit_transform(npi)
elif npi_reduce_method == 'sparse_pca':
    npi_reduce_model = SparsePCA(n_components=npi_components, alpha=0.5)
    npi = npi_reduce_model.fit_transform(npi)

setup_seed(6)
if sparsity:
    if dimension_method == 'cca':
        importr('PMA')
        r = robjects.r
        numpy2ri.activate()
        out_pls_r = r['CCA'](X_train, npi, K = ca_components, penaltyx = c_, penaltyz = 1, standardize = True)
        value = list(out_pls_r)
        name = list(out_pls_r.names)
        w_x = value[0]
        w_y = value[1]
        X_train_ = np.dot(X_train, w_x)
        y_train_ = np.dot(npi, w_y)
        X_more = np.dot(half_feature_oas_more, w_x)

    elif dimension_method == 'pls':
        importr('mixOmics')
        r = robjects.r
        numpy2ri.activate()
        c = c_*X_train.shape[-1]
        c_list = r['c'](c, c, c) 
        out_pls_r = r['spls'](X_train, npi, ncomp = ca_components, keepX = c_list)
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
        w_y = np.reshape(np.array(loading[1]), (ca_components, npi.shape[1])).T
    x_load = np.dot(X_train_.T, X_train).T / np.diagonal(np.dot(X_train_.T, X_train_))
    pctVar = sum(abs(x_load)*abs(x_load),1) / sum(sum(abs(X_train)*abs(X_train),1))
else:
    if dimension_method == 'cca':
        pls = CCA(n_components=ca_components)
        X_train_, y_train_ = pls.fit_transform(X_train, npi) 
        # pls.fit(X_train, y_train, cu = 0, cv=0) 

    elif dimension_method == 'pls':
        pls = PLSRegression(n_components=ca_components)
        X_train_, y_train_ = pls.fit_transform(X_train, npi) 
    w_x = pls.x_weights_
    w_y = pls.y_weights_

sio.savemat(r'E:\PHD\learning\research\AD_two_modal\result\revised\cca_feature.mat', {'brain_feature':X_train_, 'subject': subjects_oas_bl.values, 
                                                                                      'brain_feature_more': X_more, 'subject_more':sub_more, 'label': label_bl, 
                                                                                      'label_more': dx_more, 'sess_more': sess_more})
r_total_f = list(pearsonr(X_train_[:,i], y_train_[:,i]) for i in range(4))
print(r_total_f)
if not os.path.exists(te_save_path):
    os.makedirs(te_save_path)

if X_train_.shape[0] != continue_pcd_bl.shape[0]:
    X_train_ = X_train_[:continue_pcd_bl.shape[0]]
    y_train_ = y_train_[:continue_pcd_bl.shape[0]]
    npi_raw2 = npi_raw2[:continue_pcd_bl.shape[0]]
    half_feature = half_feature[:continue_pcd_bl.shape[0]]
    
    
#############################################################################
##########Correlates_of_brain_canonical_variates_and_other_measures
def vis1(X_train_, num_cp=3, name = 'Correlates_of_brain_canonical_variates_and_other_measures', y_label_bl=['brain comps 1', 'brain comps 2', 'brain comps 3']):
    id_null = ~np.isnan(continue_pcd_bl)

    r_matrix = np.zeros((2, num_cp, continue_pcd_bl.shape[-1]))
    for i in range(num_cp):
        for j in range(continue_pcd_bl.shape[-1]):
            subjects_oas_bl_ = subjects_oas_bl[id_null[:,j]]
            subjects_uni, subjects_uni_idx, subjects_oas_bl_idx = np.unique(subjects_oas_bl_, return_index=True, return_inverse = True)
            x = X_train_[id_null[:,j],i]
            y = continue_pcd_bl[id_null[:,j],j]
            x_list = []
            y_list = []
            for name_ in subjects_uni:
                x_list.append(np.mean(x[subjects_oas_bl_==name_]))
                y_list.append(np.mean(y[subjects_oas_bl_==name_]))
            r_total_f = pearsonr(x_list, y_list)
            r_matrix[0,i,j] = r_total_f[0]
            r_matrix[1,i,j] = r_total_f[1]
            
    fdr_correct = np.zeros((num_cp, continue_pcd_bl.shape[-1]))
    for i in range(num_cp):
        out = multitest.multipletests(r_matrix[1][i], method="fdr_bh")[1]
        fdr_correct[i] = out
    # # fdr_correct[0] = out[:len(out)//2]
    # # fdr_correct[1] = out[len(out)//2:]
    # out1 = multitest.multipletests(r_matrix[1][0], method="fdr_bh")[1]
    # out2 = multitest.multipletests(r_matrix[1][1], method="fdr_bh")[1]
    # out3 = multitest.multipletests(r_matrix[1][2], method="fdr_bh")[1]
    # mask = np.zeros((r_matrix[1][0].shape))==0
    # mask[4] = False
    # # out1_ = multitest.multipletests(r_matrix[1][0][mask], method="fdr_bh")[1]
    # # out2_ = multitest.multipletests(r_matrix[1][1][mask], method="fdr_bh")[1]
    # fdr_correct[0] = out1
    # fdr_correct[1] = out2
    # fdr_correct[2] = out3
    r_matrix_fdr = r_matrix[0]*(fdr_correct<=0.05*1)
    
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    try:
        norm = mcolors.TwoSlopeNorm(vmin=r_matrix_fdr.min(), vmax = r_matrix_fdr.max(), vcenter=0)
        im = plt.imshow(r_matrix_fdr, cmap = 'RdBu', norm=norm)
    except ValueError:
        im = plt.imshow(r_matrix_fdr, cmap = 'RdBu')
    im_ratio = r_matrix_fdr.shape[0]/r_matrix_fdr.shape[1] 
    plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    # name = 'Correlates_of_brain_canonical_variates_and_other_measures'
    plt.title(name)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.setp(ax.get_xticklabel_bls(), rotation=90, ha="right",
              rotation_mode="anchor")
    # plt.xticks(np.arange(continue_pcd_bl.shape[-1]), ['mmse', 'DIGIF', 'DIGIB',
    #                                                'ANIMALS', 'VEG', 'TRAILA', 'TRAILB', 
    #                                                'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON', 'sumbox'])   
    plt.xticks(np.arange(continue_pcd_bl.shape[-1]), ['age', 'edu', 'mmse', 'DIGIF', 'DIGIB',
                                                    'ANIMALS', 'VEG', 'TRAILA', 'TRAILB', 
                                                    'WAIS', 'LOGIMEM', 'MEMUNITS', 'BOSTON', 'sumbox',
                                                    'Centil_fBP_TOT_CORTMEAN', 'Centil_fSUVR_TOT_CORTMEAN', 
                                                    'Centil_fBP_rsf_TOT_CORTMEAN', 'Centil_fSUVR_rsf_TOT_CORTMEAN', 'GDS'])   
    # plt.xticks(np.arange(continue_pcd_bl.shape[-1]), ['Age', 'apoe', 'mmse', 'DIGIF', 'DIGIB', 'EDU',
    #                             'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB', 
    #                             'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON'])
    # plt.yticks(np.arange(num_cp), y_label_bl)
    ax.set_yticks(np.arange(num_cp))
    for i in range(r_matrix_fdr.shape[0]):
        for j in range(r_matrix_fdr.shape[1]):
            ax.text(j,i,r_matrix[0][i,j].round(2),ha="center", va="center", color="black",fontsize=20,fontname='Times New Roman')
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    
    return r_matrix
#############################################
############anova
def vis2(comp=0, variable='brain', p=False):
    subjects_uni, subjects_uni_idx, subjects_oas_bl_idx = np.unique(subjects_oas_bl, return_index=True, return_inverse = True)
    if variable == 'brain':
        array = []
        for name in subjects_uni:
            array.append(np.mean(X_train_[subjects_oas_bl==name], 0))
        # array = X_train_
        # name = 'Brain canonical variate {}'.format(comp+1)
        name = 'Brain_canonical_variate_{}'.format(comp+1)
        name1 = 'Brain canonical variate {}'.format(comp+1)

    elif variable == 'npi':
        array = []
        for name in subjects_uni:
            array.append(np.mean(y_train_[subjects_oas_bl==name]), 0)
        # array = y_train_
        name = 'NPI_canonical_variate_{}'.format(comp+1)
        name1 = 'NPI canonical variate {}'.format(comp+1)
    
    decret_pcd_bl_ = []
    for name_ in subjects_uni:
        decret_pcd_bl_.append(np.mean(decret_pcd_bl[subjects_oas_bl==name_], 0))
            
    array = np.array(array)
    decret_pcd_bl__ = np.array(decret_pcd_bl_)

    # items = ['cdr', 'SEX', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe']
    items = ['cdr', 'SEX', 'commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'apoe', 'CVHATT', 'CVAFIB', 'CVANGIO',
             'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA', 'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF', 'TRAUMEXT',
             'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO', 'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 
             'ALCOHOL', 'TOBAC30', 'TOBAC100', 'ABUSOTHR', 'PSYCDIS', 'ABRUPT', 'STEPWISE', 'SOMATIC', 'EMOT', 'HXHYPER', 'HXSTROKE', 
             'FOCLSYM', 'FOCLSIGN', 'HACHIN', 'CVDCOG', 'STROKCOG', 'CVDIMAG', 'PDNORMAL', 'SPEECH', 'FACEXP',  'TRESTFAC',
             'TRESTRHD', 'TRESTLHD', 'TRESTRFT', 'TRESTLFT', 'TRACTRHD', 'TRACTLHD','RIGDNECK', 'RIGDUPRT', 'RIGDUPLF', 'RIGDLORT', 
             'RIGDLOLF', 'TAPSRT', 'TAPSLF', 'HANDMOVR', 'HANDMOVL', 'HANDALTR', 'HANDALTL', 'LEGRT', 'LEGLF', 'ARISING', 'POSTURE', 
             'GAIT', 'POSSTAB', 'BRADYKIN', 'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
    
    for k in range(len(items)):
        if items[k] == 'SEX':
            a = 0
        if k > 1:
            idx = k+1
        else:
            idx = k
        value_cdr = []
        groupID_cdr = []
        for i, j in enumerate(list(set(decret_pcd_bl__[:,idx]))):
            if sum(decret_pcd_bl__[:,idx] == j)<10 or math.isnan(j):
                continue
            value_cdr.extend(list(array[decret_pcd_bl__[:,idx] == j, comp]))
            groupID_cdr.extend(list(np.ones(len(np.where(decret_pcd_bl__[:,idx] == j)[0])) * j))
                
        # value_cdr = value_cdr[groupID_cdr!=2]
        # groupID_cdr = groupID_cdr[groupID_cdr!=2]
        df = {'cdr_id': groupID_cdr, name: value_cdr}
        data = pd.DataFrame(df)
        # df2 = data[data['cdr_id'] != 2]
        mod = ols('{}~C(cdr_id)'.format(name),data=data).fit()
        anova_reC= anova_lm(mod)
        print(anova_reC)

        # stat, p_ = levene(value_array[0], value_array[1], value_array[2], value_array[3])
        stat, p_ = anova_reC['F'].iloc[0], anova_reC['PR(>F)'].iloc[0]
        plt.figure(figsize =(13,13))
        ax = plt.gca()
        sns.violinplot(x="cdr_id", y=name, data=data, width = 0.4, color = 'grey', inner = 'box', alpha = 0.2, ax = ax)
        plt.setp(ax.collections, alpha=.15)
        sns.stripplot(x="cdr_id", y=name, data=data, size = 8, palette="Set1", ax = ax)
        # sns.boxplot(x="cdr_id", y=name, data=data, width = 0.5, medianprops=dict(color='gray'), boxprops=dict(facecolor='gray', alpha = 0.2),
        #                capprops=dict(color='gray'),  whiskerprops=dict(color='gray'), showfliers=False, ax = ax)
        max_y = array[:,comp].max()
        min_y = array[:,comp].min()
        ax.set_yticks(np.arange(int(min_y), int(max_y+30), int((max_y-min_y)/3)))
        ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
        name_ = '{}_{}_ANOVA'.format(name, items[k])
        # plt.title(name_)
        # plt.xticks(np.arange(4), ['No impair', 'Questionable impair', 'Mild impair', 'Moderate impair'])
        # plt.xticks(np.arange(3), ['No impair', 'Questionable impair', 'Mild impair'])
        ax.set_ylabel_bl(name1, fontproperties=font2)
        ax.set_xlabel_bl(items[k], fontproperties=font2)
        plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name_)))
        
        plt.figure(figsize =(10,10))
        ax = plt.gca()
        # if not p:
        #     p = anova_reC['PR(>F)'].iloc[0]
        ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
        plt.text(0.1, 0.05, 'all P: {:.3}\n'.format(p_))
        plt.text(0.1, 0.1, 'all F: {:.3}\n'.format(stat))
        if len(set(groupID_cdr))>2:
            tukey = pairwise_tukeyhsd(endog=df[name],
                              groups=df['cdr_id'],
                              alpha=0.05)
            out = tukey.summary().data
            import scikit_posthocs as sp
            out2 = sp.posthoc_ttest(data, val_col=name, group_col='cdr_id', p_adjust=None, pool_sd=True)
            out2 = np.expand_dims(out2.values,0)
            out2_half = keep_triangle_half(out2.shape[1] * (out2.shape[1]-1)//2, out2.shape[0], out2).squeeze()

    # t, p = ttest_ind(data[name][data['cdr_id']==0], data[name][data['cdr_id']==1])

            for i in range(1,len(out)):
                plt.text(0.1, 0.1*i+0.15, '{} vs {} P: {:.3} {}\n'.format(out[i][0], out[i][1], out[i][3], out2_half[i-1]))
                plt.text(0.1, 0.1*i+0.2, '{} vs {} meandiff: {:.3}\n'.format(out[i][0], out[i][1], out[i][2]))
        plt.savefig(os.path.join(te_save_path, '{}_stat.svg'.format(name_)))

    return anova_reC['F'].iloc[0], anova_reC['PR(>F)'].iloc[0]

############pca_cca_yloadings and correlation
def vis3(comp=0, array = None, name2 = 'none'):
    font2 = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 35}
    matplotlib.rc('font', **font2)

    # weight_fc = fmri_pca.components_.T
    weight_npi= npi_reduce_model.components_.T
    w_npi = np.dot(npi_reduce_model.components_.T,w_y)
    # w_npi = w_y
    plt.figure(figsize =(18,18))
    ax = plt.gca()
    try:
        norm = mcolors.TwoSlopeNorm(vmin=w_npi.min(), vmax = w_npi.max(), vcenter=0)
        im = plt.imshow(w_npi, cmap = 'RdBu_r', norm=norm)
    except ValueError:
        im = plt.imshow(w_npi, cmap = 'RdBu_r')
    im_ratio = w_npi.shape[0]/w_npi.shape[1] 
    # plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    plt.colorbar()
    name = 'pca_cca_yloadings'
    plt.title(name)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['bottom'].set_color('none')  # 设置上‘脊梁’为无色
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.yticks(np.arange(12), ['Delusions', 'Hallucinations', 'Agitation', 'Depression', 'Anxiety', 'Euphoria', 
                  'Apathy', 'Disinhibition', 'Irritability', 'Aberrant motor\nbehavior', 'Nighttime behavior\ndisturbances', 'Appetite abnormalities'])   
    plt.xticks(np.arange(4), ['Affective mode', 'Psychotic mode', 'other mode', 'other mode'])
    # ax.axes.get_xaxis().set_visible(False) 
    for i in range(w_npi.shape[0]):
        for j in range(w_npi.shape[1]):
            text = ax.text(j,i,w_npi[i,j].round(2),ha="center", va="center", color="black",fontsize=20,fontname='Times New Roman')
    # plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight') 
    ###################################################################################
    # if variable == 'brain':
    #     # array = X_train_
    #     # name = 'Brain canonical variate {}'.format(comp+1)
    #     name = 'Brain_component{}_NPI_correlation_spear'.format(comp+1)
    #     # name1 = 'Brain canonical variate {}'.format(comp+1)

    # elif variable == 'npi':
    #     # array = y_train_
    #     name = 'NPI_component{}_NPI_correlation_spear'.format(comp+1)
    #     # name1 = 'NPI canonical variate {}'.format(comp+1)
    name = name2
    loading_r = np.zeros((array.shape[-1], npi_raw2.shape[-1]))
    loading_p = np.zeros((array.shape[-1], npi_raw2.shape[-1]))
    for i in range(array.shape[-1]):
        for j in range(npi_raw2.shape[-1]): 
            # r_poly = r['polyserial'](array[:,j], npi[:,i])
            # r_poly = pearsonr(npi_raw2[:,j], array[:,i])
            r_poly = spearmanr(npi_raw2[:,j], array[:,i])
            
            r_ = list(r_poly)[0]
            loading_r[i,j] = r_
            loading_p[i,j] = list(r_poly)[1]
            
    plt.figure(figsize =(10,16))
    ax = plt.gca()
    data = pd.DataFrame({'x tick':np.arange(loading_r.shape[-1]), 'correlation':loading_r[comp]})
    data['sign'] = data['correlation'] > 0
    data['correlation'].plot(kind='barh', color=data.sign.map({True: (1.0, 0, 0, 0.7), False: '#87CEFA'}), ax=ax)
    # data.plot(y = 'correlation', kind='bar', color=data.sign.map({True: (1.0, 0, 0, 0.7), False: '#87CEFA'}), ax=ax)
    ax.axvline(0, color='black')
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    max_y = data['correlation'].max()
    min_y = data['correlation'].min()
    ax.set_xticks(np.around(np.arange(round(min_y,2), round(max_y, 2), 0.1), 2))
    ax.set_xlabel_bl('R', fontproperties=font2)

    plt.setp(ax.get_yticklabel_bls(), rotation=30, ha="right",
              rotation_mode="anchor")
    plt.yticks(np.arange(12), ['Delusions', 'Hallucinations', 'Agitation', 'Depression', 'Anxiety', 'Euphoria', 
                  'Apathy', 'Disinhibition', 'Irritability', 'Aberrant motor behavior', 'Nighttime behavior disturbances', 'Appetite abnormalities'])   
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    # plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
    # plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))

    plt.close('all')
    return loading_p
    
################t test sex
def vis4(comp=0, variable='brain'):
    if variable == 'brain':
        array = X_train_
        name = 'Brain canonical variate {}'.format(comp+1)
    elif variable == 'npi':
        array = y_train_
        name = 'NPI canonical variate {}'.format(comp+1)
    value_array = []
    id_null = ~np.isnan(decret_pcd_bl)
    decret_pcd_bl_2 = decret_pcd_bl[id_null[:,1]]
    X_train_2 = array[id_null[:,1]]
    for i, j in enumerate(list(set(decret_pcd_bl_2[:,1]))):
        if i == 0:
            value_cdr = X_train_2[decret_pcd_bl_2[:,1] == j, comp]
            groupID_cdr = np.ones(len(np.where(decret_pcd_bl_2[:,1] == j)[0])) * j 
            value_array.append(X_train_2[decret_pcd_bl_2[:,1] == j, comp])
        else:
            value_cdr = np.concatenate((value_cdr, X_train_2[decret_pcd_bl_2[:,1] == j, comp]))
            groupID_cdr = np.concatenate((groupID_cdr, np.ones(len(np.where(decret_pcd_bl_2[:,1] == j)[0])) * j ))
            value_array.append(X_train_2[decret_pcd_bl_2[:,1] == j, comp])
    t, p = ttest_ind(value_array[0], value_array[1])
    df = {'sex': groupID_cdr, name: value_cdr}
    data = pd.DataFrame(df)
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    sns.stripplot(x="sex", y=name, data=data, size = 8, palette="Set2", ax = ax)
    sns.boxplot(x="sex", y=name, data=data, width = 0.5, medianprops=dict(color='gray'), boxprops=dict(facecolor='gray', alpha = 0.2),
                   capprops=dict(color='gray'),  whiskerprops=dict(color='gray'), showfliers=False, ax = ax)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    plt.text(1, -2, 'P: {:.3}\n'.format(p))
    name = '{}_sex_t-test'.format(name)
    plt.title(name)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    return t, p
###################################################################################
##########npi score
def vis5():
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    norm = mcolors.TwoSlopeNorm(vmin=npi.min()-0.1, vmax = npi.max(), vcenter=0)
    im = plt.imshow(npi, cmap = 'RdBu', norm=norm)
    im_ratio = npi.shape[0]/npi.shape[1] 
    # plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    plt.colorbar()
    name = 'npi_after{}mapping'.format(npi_reduce_method)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.title(name)
    
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    plt.close('all')

############################################################
###############brain pca pls loading
def vis6(comp=0, variable = 'roi', vis = 'full', decomp = True, roi_num = 100, fc_reduce_method2 = 'pca'):
    if vis == 'full':
        name = 'brain_pca_pls_loading_comp{}_{}_full'.format(comp, variable)
    else:
        name = 'brain_pca_pls_loading_comp{}_{}_top100'.format(comp, variable)
    if decomp:
        if fc_reduce_method2 == 'mad':
            fc_pca_pls_loading = np.zeros((roi_num*(roi_num-1)//2, ca_components))
            fc_pca_pls_loading[mask] = w_x
        else:
            fc_pca_pls_loading = np.dot(fmri_pca.components_.T, w_x)
        fc_pca_pls_loading_sys, _ = vector_to_matrix(fc_pca_pls_loading[:,comp], roi_num)
    else:
        fc_pca_pls_loading_sys, _ = vector_to_matrix(w_x[:,comp], roi_num)
    if roi_num == 100:
        atlas = 'shf_100'
    else:
        atlas = 'shf_200'

    if variable == 'roi':
        if vis != 'full':
            if decomp:
                fc_pca_pls_loading_sys[abs(fc_pca_pls_loading_sys)<np.sort(abs(fc_pca_pls_loading[:,comp]))[-100]] = 0
            else:
                fc_pca_pls_loading_sys[abs(fc_pca_pls_loading_sys)<np.sort(abs(w_x[:,comp]))[-100]] = 0
        fig, ax = plt.subplots(figsize=(15, 15))
        roi_idx = np.arange(0,roi_num,1)
        im, cbar = heatmap(fc_pca_pls_loading_sys, roi_idx, ax=ax, cmap="RdBu_r", 
                            cbarlabel="Weights",atlas = atlas, half_or_full='half') 
        # im, cbar = heatmap(fc_pca_pls_loading_sys, roi_idx, ax=ax, cmap="RdBu_r", 
        #                     cbarlabel_bl="Weights") 
        # plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
    else:
        net_pca_pls_loading_sys = get_net_net_connect(np.expand_dims(abs(fc_pca_pls_loading_sys), 0), 1)
        net_pca_pls_loading_sys = net_pca_pls_loading_sys.squeeze()
        vis_loading_sys =net_pca_pls_loading_sys
        mask_id =  abs(np.tri(vis_loading_sys.shape[0], k=0)-1)
        vis_loading_sys = np.ma.array(vis_loading_sys, mask=mask_id) # mask out the lower triangle
        fig, ax = plt.subplots(figsize=(15, 15))
        net_idx = np.arange(0,7,1)
        im, cbar = heatmap(vis_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type='net',
                            cbarlabel="Weights", dash_line = 'no',atlas = atlas) 
        # im, cbar = heatmap(vis_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type='net',
        #                     cbarlabel_bl="Weights", dash_line = 'no') 
        # plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
        
    return fc_pca_pls_loading_sys
###############################################################
##########npi and brain transformed score
def vis7(comp=0):
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    norm = mcolors.TwoSlopeNorm(vmin=X_train_.min(), vmax = X_train_.max(), vcenter=0)
    im = plt.imshow(X_train_, cmap = 'RdBu_r', norm=norm)
    im_ratio = npi.shape[0]/npi.shape[1] 
    plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    # plt.colorbar()1
    name = 'cca_X_train_transform'
    plt.title(name)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
    
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    norm = mcolors.TwoSlopeNorm(vmin=y_train_.min(), vmax = y_train_.max(), vcenter=0)
    im = plt.imshow(y_train_, cmap = 'RdBu_r', norm=norm)
    im_ratio = npi.shape[0]/npi.shape[1] 
    plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    name = 'cca_y_train_transform'
    plt.title(name)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
###############################################################
########transformed training x y scatter
def vis8(comp=0):
    if comp == 0:
        color = 'r'
    else:
        color = '#87CEFA'
    plt.figure(figsize =(10,10))
    ax = plt.gca()
    idx = np.where(X_train_[:,comp]!=X_train_[:,comp].min())[0]
    plt.scatter(X_train_[idx,comp], y_train_[idx,comp], alpha=.6, color = color)
    name = 'tr_cca_comp{}_xy'.format(comp)
    plt.title(name)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    max_x = X_train_[:,comp].max()
    min_x = X_train_[:,comp].min()
    max_y = y_train_[:,comp].max()
    min_y = y_train_[:,comp].min()
    plt.text(abs(max_x - min_x)*0.8+min_x,min_y+abs(max_y - min_y)*0.01, 'R: {:.3}\nP: {:.3}\n'.format(r_total_f[comp][0], r_total_f[comp][1]))
    # ax.set_xticks(np.around(np.arange(min_x, max_x, abs(max_x - min_x)//3), 1))
    # ax.set_yticks(np.around(np.arange(min_y, max_y, abs(max_y - min_y)//3), 1))
    ax.set_xlabel_bl('Function correlation canonical variates', fontproperties=font)
    ax.set_ylabel_bl('NPI canonical variates', fontproperties=font)  
    # ax.set_xlabel_bl('NPI canonical variates', fontproperties=font)
    # ax.set_ylabel_bl('Function correlation canonical variates', fontproperties=font)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name))) 

########################################################
############transformed test x y scatter
def vis9(comp, X_te_, y_te_, r, p, name = False):
    if comp == 0:
        color = 'r'
    elif comp == 1:
        color = '#87CEFA'
    else:
        color = 'grey'
    plt.figure(figsize =(10,10))
    ax = plt.gca()
    plt.scatter(X_te_[:,comp], y_te_[:,comp],alpha=.6, color = color)
    plt.title(name)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    max_x = X_te_[:,comp].max()
    min_x = X_te_[:,comp].min()
    max_y = y_te_[:,comp].max()
    min_y = y_te_[:,comp].min()
    plt.text(abs(max_x - min_x)*0.8+min_x,min_y+abs(max_y - min_y)*0.01,  'R: {:.3}\nP: {:.3}\n'.format(r, p))
    ax.set_xticks(np.around(np.arange(min_x, max_x, round(abs(max_x - min_x)/3)), 0))
    ax.set_yticks(np.around(np.arange(min_y, max_y, round(abs(max_y - min_y)/3)), 0))
    # ax.set_xlabel_bl('NPI canonical variates', fontproperties=font)
    # ax.set_ylabel_bl('Function correlation canonical variates', fontproperties=font)
    ax.set_xlabel_bl('FC canonical variates', fontproperties=font2)
    ax.set_ylabel_bl('NPS canonical variates', fontproperties=font2)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))    
############################################################
############brain composite and fc correlation top100
def vis10(comp=0, vis = 'top', decomp = True, roi_num = 100, fc_reduce_method2 = 'pca'):
    if vis == 'full':
        name = 'brain_pls_score_comp{}_&_fc_corr'.format(comp)
    elif vis == 'top':
        name = 'brain_pls_score_comp{}_&_fc_corr_fdr_top100'.format(comp)
    elif vis == 'fdr':
        name = 'brain_pls_score_comp{}_&_fc_corr_fdr'.format(comp)
    if decomp:
        if fc_reduce_method2 == 'mad':
            fc_pca_pls_loading = np.zeros((roi_num*(roi_num-1)//2, ca_components))
            fc_pca_pls_loading[mask] = w_x
        else:
            fc_pca_pls_loading = np.dot(fmri_pca.components_.T, w_x)

        weight =  fc_pca_pls_loading[:,comp]
    else:
        weight = w_x[:,comp]
    r_matrix = np.zeros((half_feature.shape[-1],2))
    score1 = X_train_[:,comp]
    for i in range(half_feature.shape[-1]):
        if weight[i] != 0: 
            fc = half_feature[:,i]
            # out = pearsonr(fc, score1)
            out = spearmanr(fc, score1)
            r_matrix[i,0] = out[0]
            r_matrix[i,1] = out[1]
        
    p = multitest.fdrcorrection(r_matrix[weight != 0,-1])
    if vis == 'full':
        fc_pca_pls_loading = r_matrix[:,0]
    elif vis == 'fdr':
        fc_pca_pls_loading = r_matrix[:,0]
        fc_pca_pls_loading[weight != 0] = fc_pca_pls_loading[weight != 0] * (p[0] * 1)
        fc_pca_pls_loading[weight == 0] = 0
    elif vis == 'top':
        fc_pca_pls_loading = r_matrix[:,0]
    if roi_num == 100:
        atlas = 'shf_100'
    else:
        atlas = 'shf_200'
    fc_pca_pls_loading_sys, _ = vector_to_matrix(fc_pca_pls_loading, roi_num)
    # fc_pca_pls_loading_sys, _ = vector_to_matrix(fc_pca_pls_loading, 100)
    # mask_id =  abs(np.tri(fc_pca_pls_loading_sys.shape[0], k=0)-1)
    if vis == 'top':
        fc_pca_pls_loading_sys[abs(fc_pca_pls_loading_sys)<np.sort(abs(fc_pca_pls_loading))[-100]] = 0

    # fc_pca_pls_loading_sys = np.ma.array(fc_pca_pls_loading_sys, mask=mask_id) # mask out the lower triangle
    # RdBu_r
    fig, ax = plt.subplots(figsize=(15, 15))
    roi_idx = np.arange(0,100,1)
    im, cbar = heatmap(fc_pca_pls_loading_sys, roi_idx, ax=ax, cmap="RdBu_r", 
                        cbarlabel_bl="Correlation coefficient", atlas = atlas) 

    plt.title(name)
    plt.savefig(os.path.join(te_save_path, '{}_spear.svg'.format(name)), bbox_inches = 'tight')
    sio.savemat(os.path.join(te_save_path, 'corr_comp_{}_{}_spear.mat'.format(comp, vis)), {'corr': fc_pca_pls_loading_sys})

    return fc_pca_pls_loading_sys
############################################################
############brain composite and ROI strength correlation top15
def vis11(fc, comp=0):
    fc = fc[idx,:]   
    fc = np.mean(abs(fc), -1)
    score1 = X_train_[:,comp]
    loading_r = np.zeros((fc.shape[-1], 2))
    for i in range(fc.shape[-1]):
        r, p = pearsonr(fc[:,i], score1)
        loading_r[i,0] = r
        loading_r[i,1] = p
    p_ = multitest.fdrcorrection(loading_r[:,-1])
    roi_label_bls = get_rois_label_bl()
    roi_label_bls = np.array(roi_label_bls)
    roi_label_bls = roi_label_bls[p_[0]]
    loading_r = loading_r[p_[0]]
    
    idx_ = np.argsort(-loading_r[:,0])[:15]
    loading_r = loading_r[idx_]
    roi_label_bls = roi_label_bls[idx_]
    plt.figure(figsize =(16,16))
    ax = plt.gca()
    data = pd.DataFrame({'correlation':loading_r[:,0]})
    data['sign'] = data['correlation'] > 0
    data['correlation'].plot(kind='bar', color=data.sign.map({True: (1.0, 0, 0, 0.7), False: '#87CEFA'}), ax=ax)
    ax.axhline(0, color='black')
    name = 'brain_component{}_fc_ROI_strength_correlation_top15'.format(comp)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.title(name)
    plt.xticks(np.arange(len(roi_label_bls)), roi_label_bls)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    plt.close('all')

###########################################################################
##pca npi weight
def vis12():
    weight_npi= npi_reduce_model.components_.T
    w_npi = weight_npi
    plt.figure(figsize =(20,20))
    ax = plt.gca()
    try:
        norm = mcolors.TwoSlopeNorm(vmin=w_npi.min(), vmax = w_npi.max(), vcenter=0)
        im = plt.imshow(w_npi, cmap = 'RdBu_r', norm=norm)
    except ValueError:
        im = plt.imshow(w_npi, cmap = 'RdBu_r')
    im_ratio = w_npi.shape[0]/w_npi.shape[1] 
    # plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    plt.colorbar()
    name = 'pca_npi_weights'
    plt.title(name)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['bottom'].set_color('none')  # 设置上‘脊梁’为无色
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.yticks(np.arange(12), ['Delusions', 'Hallucinations', 'Agitation', 'Depression', 'Anxiety', 'Euphoria', 
                  'Apathy', 'Disinhibition', 'Irritability', 'Aberrant motor\nbehavior', 'Nighttime behavior\ndisturbances', 'Appetite abnormalities'])   
    ax.axes.get_xaxis().set_visible(False) 
    for i in range(w_npi.shape[0]):
        for j in range(w_npi.shape[1]):
            text = ax.text(j,i,w_npi[i,j].round(2),ha="center", va="center", color="black",fontsize=20,fontname='Times New Roman')
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight') 
    
##cca npi weight
def vis13():
    w_npi = w_y
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    try:
        norm = mcolors.TwoSlopeNorm(vmin=w_npi.min(), vmax = w_npi.max(), vcenter=0)
        im = plt.imshow(w_npi, cmap = 'RdBu_r', norm=norm)
    except ValueError:
        im = plt.imshow(w_npi, cmap = 'RdBu_r')
    im_ratio = w_npi.shape[0]/w_npi.shape[1] 
    # plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    plt.colorbar()
    name = 'pls_npi_weights'
    plt.title(name)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['bottom'].set_color('none')  # 设置上‘脊梁’为无色
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.axes.get_xaxis().set_visible(False) 
    ax.axes.get_yaxis().set_visible(False) 
    for i in range(w_npi.shape[0]):
        for j in range(w_npi.shape[1]):
            text = ax.text(j,i,w_npi[i,j].round(2),ha="center", va="center", color="black",fontsize=20,fontname='Times New Roman')
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight') 
################t test diagnosis label_bl
def vis14(comp=0, variable='brain', p=False):
    if variable == 'brain':
        array = X_train_
        name = 'brain_comp{}_diagnosis_t-test'.format(comp)
        y_name = "Brain canonical variate {}".format(comp+1)
    elif variable == 'npi':
        array = y_train_
        name = 'npi_comp{}_diagnosis_t-test'.format(comp)
        y_name = "NPI canonical variate {}".format(comp+1)

    HC_mask = label_bl == 'Cognitively normal'
    AD_mask = (label_bl != 'Cognitively normal') & (label_bl != 'uncertain dementia')
    label_bl[AD_mask] = 'Dementia'
    groupID = np.concatenate((label_bl[HC_mask], label_bl[AD_mask]))
    value = np.concatenate((array[HC_mask, comp], array[AD_mask, comp]))
    t, p_ = ttest_ind(array[HC_mask,comp], array[AD_mask,comp])
    df = {'Diagnosis label_bl': groupID, y_name: value}
    data = pd.DataFrame(df)
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    my_pal = {"Cognitively normal": "r", "Dementia": "#87CEFA"}
    sns.violinplot(x="Diagnosis label_bl", y=y_name, data=data, width = 0.3, color = 'grey', inner = 'box', alpha = 0.1, ax = ax)
    plt.setp(ax.collections, alpha=.15)
    sns.stripplot(x="Diagnosis label_bl", y=y_name, data=data, size = 8, palette='Set1', ax = ax)
        # sns.stripplot(x="cdr_id", y=name, data=data, size = 8, palette="Set1", ax = ax)

    # sns.boxplot(x="Diagnosis label_bl", y=y_name, data=data, width = 0.4, medianprops=dict(color='gray'), boxprops=dict(facecolor='gray', alpha = 0.2),
    #                capprops=dict(color='gray'),  whiskerprops=dict(color='gray'), showfliers=False, ax = ax)

    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    max_y = array[:,comp].max()
    min_y = array[:,comp].min()
    if not p:
        p = p_
    plt.text(1.2, abs(max_y - min_y)*0.01+min_y, 'P: {:.3}\n'.format(p))
    plt.text(1.2, -4*abs(max_y - min_y)*0.01+min_y, 't: {:.3}\n'.format(t))
    ax.set_xlabel_bl("Diagnosis label_bl", fontproperties=font2)
    ax.set_ylabel_bl(y_name, fontproperties=font2)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    return t, p
###################################################################################
###############difference in anxiety and depression exploration 
####difference in anxiety and depression of 2 comps 
def vis15(data, comp=0, variable = 'npi', p=False, domain = 'Depression'):

    if domain == 'Depression':
        data = data[:,3]
    elif domain == 'Anxiety':
        data = data[:,4]
    if variable == 'brain':
        array = X_train_
        name = 'Brain_canonical_variate_{}'.format(comp+1)
        name1 = 'Brain canonical variate {}'.format(comp+1)

    elif variable == 'npi':
        array = y_train_
        name = 'NPI_canonical_variate_{}'.format(comp+1)
        name1 = 'NPI canonical variate {}'.format(comp+1)
        
    value_array = []
    for i, j in enumerate(list(set(data))):
        if i == 0:
            value_cdr = array[data == j, comp]
            groupID_cdr = np.ones(len(np.where(data == j)[0])) * j 
            value_array.append(array[data == j, comp])
        else:
            value_cdr = np.concatenate((value_cdr, array[data == j, comp]))
            groupID_cdr = np.concatenate((groupID_cdr, np.ones(len(np.where(data == j)[0])) * j ))
            value_array.append(array[data == j, comp])
    
    df = {domain: groupID_cdr, name: value_cdr}
    # df2 = df[df['cdr_id'] != 2]
    data = pd.DataFrame(df)
    # df2 = data[data['cdr_id'] != 2]
    anova_reC= anova_lm(ols('{}~C({})'.format(name,domain),data=data).fit())
    print(anova_reC)
    # stat, p_ = levene(value_array[0], value_array[1], value_array[2], value_array[3])
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    sns.stripplot(x=domain, y=name, data=data, size = 8, palette="Set2", ax = ax)
    sns.boxplot(x=domain, y=name, data=data, width = 0.5, medianprops=dict(color='gray'), boxprops=dict(facecolor='gray', alpha = 0.2),
                   capprops=dict(color='gray'),  whiskerprops=dict(color='gray'), showfliers=False, ax = ax)
    max_y = array[:,comp].max()
    min_y = array[:,comp].min()
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    if not p:
        p = anova_reC['PR(>F)'].iloc[0]
    plt.text(3, min_y+abs(max_y - min_y)*0.01, 'P: {:.3}\n'.format(p))
    name = '{}_{}_ANOVA'.format(name, domain)
    plt.title(name)
    # plt.xticks(np.arange(4), ['No impair', 'Questionable impair', 'Mild impair', 'Moderate impair'])
    ax.set_ylabel_bl(name1, fontproperties=font)
    ax.set_xlabel_bl(domain, fontproperties=font)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    return anova_reC['F'].iloc[0], anova_reC['PR(>F)'].iloc[0]

####difference in anxiety and depression of 2 comps 
def vis16(data, comp=0, variable = 'npi', p=False, domain = 'Depression'):
    if domain == 'Depression':
        data = data[:,3]
    elif domain == 'Anxiety':
        data = data[:,4]
    array = continue_pcd[:]
    # name = 'Brain canonical variate {}'.format(comp+1)
    name = '{}'.format(variable)
    id_null = ~np.isnan(array[:,comp])
    array = array[id_null]
    data = data[id_null]
    # idx2 = np.where(array[:,comp]<60)[0]
    # array = array[idx2]
    # data = data[idx2]
    value_array = []
    for i, j in enumerate(list(set(data))):
        if i == 0:
            value_cdr = array[data == j, comp]
            groupID_cdr = np.ones(len(np.where(data == j)[0])) * j 
            value_array.append(array[data == j, comp])
        else:
            value_cdr = np.concatenate((value_cdr, array[data == j, comp]))
            groupID_cdr = np.concatenate((groupID_cdr, np.ones(len(np.where(data == j)[0])) * j ))
            value_array.append(array[data == j, comp])
    
    df = {domain: groupID_cdr, name: value_cdr}
    # df2 = df[df['cdr_id'] != 2]
    data = pd.DataFrame(df)
    # df2 = data[data['cdr_id'] != 2]
    anova_reC= anova_lm(ols('{}~C({})'.format(name,domain),data=data).fit())
    print(anova_reC)
    # stat, p_ = levene(value_array[0], value_array[1], value_array[2], value_array[3])
    plt.figure(figsize =(15,15))
    ax = plt.gca()
    sns.stripplot(x=domain, y=name, data=data, size = 8, palette="Set2", ax = ax)
    sns.boxplot(x=domain, y=name, data=data, width = 0.5, medianprops=dict(color='gray'), boxprops=dict(facecolor='gray', alpha = 0.2),
                   capprops=dict(color='gray'),  whiskerprops=dict(color='gray'), showfliers=False, ax = ax)
    max_y = array[:,comp].max()
    min_y = array[:,comp].min()
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    if not p:
        p = anova_reC['PR(>F)'].iloc[0]
    if domain == 'Depression':
        plt.text(2, min_y+abs(max_y - min_y)*0.01, 'P: {:.3}\n'.format(p))
    elif domain == 'Anxiety':
        plt.text(3, min_y+abs(max_y - min_y)*0.01, 'P: {:.3}\n'.format(p))
    name = '{} {} ANOVA'.format(name, domain)
    plt.title(name)
    # plt.xticks(np.arange(4), ['No impair', 'Questionable impair', 'Mild impair', 'Moderate impair'])
    ax.set_ylabel_bl(name, fontproperties=font)
    ax.set_xlabel_bl(domain, fontproperties=font)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    
    return anova_reC['F'].iloc[0], anova_reC['PR(>F)'].iloc[0]
##########################################
###########################################################net correlation
def vis17(fc, comp=0, method='corr_mean', vis = 'fdr', roi_num = 100):
    net_idx = np.arange(0,7,1)
    if roi_num == 100:
        atlas = 'shf_100'
    else:
        atlas = 'shf_200'
    if method == 'net_fc':
        fc = fc[idx,:]   
        fc = get_net_net_connect(fc, fc.shape[0])
        # fc = keep_triangle_half(fc.shape[1] * (fc.shape[1]-1)//2, fc.shape[0], fc)
        score1 = X_train_[:,comp]
        loading_r = np.zeros((fc.shape[-1], fc.shape[-1], 2))
        for i in range(fc.shape[-1]):
            for j in range(fc.shape[-1]):
                r, p = pearsonr(fc[:,i,j], score1)
                loading_r[i,j,0] = r
                loading_r[i,j,1] = p
        idx_ = np.tril_indices_from(loading_r[:, :, 0], 0)
        p_ = multitest.fdrcorrection(loading_r[:,:,-1][idx_])
        p_fdr, _ = vector_to_matrix(p_[0], 7, idx_)
        
        plt.figure(figsize =(15,15))
        ax = plt.gca()
        im, cbar = heatmap(loading_r[:,:,0]*p_fdr, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
                            cbarlabel_bl="Correlation coefficient",dash_line = 'none', atlas = atlas) 
        # im, cbar = heatmap(loading_r[:,:,0]*p_fdr, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
        #                     cbarlabel_bl="Correlation coefficient",dash_line = 'none') 
        name = 'net correlation comp {}'.format(comp)
        plt.title(name)
        ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
        plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
        plt.close('all')
    elif method == 'corr_mean':
        if vis == 'full':
            name = 'brain_comp{}_&_net_corr_full'.format(comp)
        elif vis == 'top':
            name = 'brain_comp{}_&_net_corr_fdr_top100'.format(comp)
        elif vis == 'fdr':
            name = 'brain_comp{}_&_net_corr_fdr'.format(comp)
        # fc_pca_pls_loading = np.dot(fmri_pca.components_.T, w_x)
        r_matrix = np.zeros((half_feature.shape[-1],2))
        score1 = X_train_[:,comp]
        for i in range(half_feature.shape[-1]):
            fc = half_feature[:,i]
            out = pearsonr(fc, score1)
            r_matrix[i,0] = out[0]
            r_matrix[i,1] = out[1]
            
        p = multitest.fdrcorrection(r_matrix[:,-1])
        if vis == 'fdr':
            fc_pca_pls_loading = r_matrix[:,0] * (p[0] * 1)
        else:
            fc_pca_pls_loading = r_matrix[:,0]
        fc_pca_pls_loading_sys, _ = vector_to_matrix(fc_pca_pls_loading, roi_num)
        # fc_pca_pls_loading_sys, _ = vector_to_matrix(fc_pca_pls_loading, 100)
        # mask_id =  abs(np.tri(fc_pca_pls_loading_sys.shape[0], k=0)-1)
        if vis == 'top':
            fc_pca_pls_loading_sys[abs(fc_pca_pls_loading_sys)<np.sort(abs(fc_pca_pls_loading))[-100]] = 0
            
        net_pca_pls_loading_sys = get_net_net_connect(np.expand_dims(abs(fc_pca_pls_loading_sys), 0), 1)
        net_pca_pls_loading_sys = net_pca_pls_loading_sys.squeeze()
        fig, ax = plt.subplots(figsize=(15, 15))
        mask_id =  abs(np.tri(net_pca_pls_loading_sys.shape[0], k=0)-1)
        net_pca_pls_loading_sys = np.ma.array(net_pca_pls_loading_sys, mask=mask_id) # mask out the lower triangle
        im, cbar = heatmap(net_pca_pls_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
                            cbarlabel_bl="Correlation coefficient (abs)",dash_line = 'noe', atlas = atlas) 
        # im, cbar = heatmap(net_pca_pls_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
        #                     cbarlabel_bl="Correlation coefficient (abs)",dash_line = 'none') 
        plt.title(name)
        plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
##########################################
###########################################################net or roi intersection correlation
def vis18(fc, variable = 'roi', roi_num = 100):
    if roi_num == 100:
        atlas = 'shf_100'
    else:
        atlas = 'shf_200'
    if variable == 'roi':
        name = 'brain_roi_intersection'
    else:
        name = 'brain_net_intersection'
    # fc_pca_pls_loading = np.dot(fmri_pca.components_.T, w_x)
    r_matrix_0 = np.zeros((half_feature.shape[-1],2))
    r_matrix_1 = np.zeros((half_feature.shape[-1],2))
    score0 = X_train_[:,0]
    score1 = X_train_[:,1]
    for i in range(half_feature.shape[-1]):
        fc = half_feature[:,i]
        out = pearsonr(fc, score0)
        r_matrix_0[i,0] = out[0]
        r_matrix_0[i,1] = out[1]       
        out = pearsonr(fc, score1)
        r_matrix_1[i,0] = out[0]
        r_matrix_1[i,1] = out[1]
        
    p1 = multitest.fdrcorrection(r_matrix_0[:,-1])
    p2 = multitest.fdrcorrection(r_matrix_1[:,-1])
    p = (p1[0]*1)*(p2[0]*1)
    fc_pca_pls_loading_0 = r_matrix_0[:,0] * p 
    fc_pca_pls_loading_1 = r_matrix_1[:,0] * p 
    fc_pca_pls_loading = (abs(fc_pca_pls_loading_0) + abs(fc_pca_pls_loading_1))/2
    fc_pca_pls_loading_sys, _ = (fc_pca_pls_loading)
    # mask_id =  abs(np.tri(fc_pca_pls_loading_sys.shape[0], k=0)-1)
    # if variable == 'top':
    #     fc_pca_pls_loading_sys[abs(fc_pca_pls_loading_sys)<np.sort(abs(fc_pca_pls_loading))[-100]] = 0
    vis_loading_sys = fc_pca_pls_loading_sys
    if variable == 'net':
        net_pca_pls_loading_sys = get_net_net_connect(np.expand_dims(abs(fc_pca_pls_loading_sys), 0), 1)
        net_pca_pls_loading_sys = net_pca_pls_loading_sys.squeeze()
        vis_loading_sys =net_pca_pls_loading_sys
        mask_id =  abs(np.tri(vis_loading_sys.shape[0], k=0)-1)
        vis_loading_sys = np.ma.array(vis_loading_sys, mask=mask_id) # mask out the lower triangle
    fig, ax = plt.subplots(figsize=(15, 15))
    if variable == 'net':
        net_idx = np.arange(0,7,1)
        # im, cbar = heatmap(vis_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
        #                     cbarlabel_bl="Correlation coefficient (abs)",dash_line = 'none') 
        im, cbar = heatmap(vis_loading_sys, net_idx, ax=ax, cmap="Reds", connect_type = 'net',
                            cbarlabel_bl="Correlation coefficient (abs)",dash_line = 'none', atlas = atlas) 
    elif variable == 'roi':
        roi_idx = np.arange(0,100,1)
        im, cbar = heatmap(vis_loading_sys, roi_idx, ax=ax, cmap="Reds", 
                            cbarlabel_bl="FC", atlas = atlas) 
        # im, cbar = heatmap(vis_loading_sys, roi_idx, ax=ax, cmap="Reds", 
        #                     cbarlabel_bl="FC") 
    plt.title(name)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)), bbox_inches = 'tight')
############################################################
############brain composite and ROI strength correlation top15 sum by fc correlation
def vis19(fc, comp=0):
    name = 'brain_comp{}_ROI_strenght_corrabs_full_top15'.format(comp)
    # fc_pca_pls_loading = np.dot(fmri_pca.components_.T, w_x)
    r_matrix = np.zeros((half_feature.shape[-1],2))
    score1 = X_train_[:,comp]
    for i in range(half_feature.shape[-1]):
        fc = half_feature[:,i]
        out = pearsonr(fc, score1)
        r_matrix[i,0] = out[0]
        r_matrix[i,1] = out[1]
        
    fc_pca_pls_loading_sys, _ = vector_to_matrix(r_matrix[:,0], roi_num)
    roi_loading_sys = np.mean(abs(fc_pca_pls_loading_sys), -1)
    idx_ = np.argsort(-roi_loading_sys)[:15]
    roi_label_bls = get_rois_label_bl()
    roi_label_bls = [line.replace('_', ' ') for line in roi_label_bls]
    roi_label_bls = np.array(roi_label_bls)
    roi_loading_sys = roi_loading_sys[idx_]
    roi_label_bls = roi_label_bls[idx_]
    plt.figure(figsize =(20,20))
    ax = plt.gca()
    data = pd.DataFrame({'Correlation (abs)':roi_loading_sys})
    data['sign'] = data['Correlation (abs)'] > 0
    data['Correlation (abs)'].plot(kind='bar', color=data.sign.map({True: (1.0, 0, 0, 0.7), False: '#87CEFA'}), ax=ax)
    ax.axhline(0, color='black')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.title(name)
    for i, roi_name in enumerate(roi_label_bls):
        if len(roi_name)>12:
            name_list = roi_name.split(' ')
            roi_name = name_list[0]+' '+name_list[1]+'\n' + name_list[2] + ' ' + name_list[3]
            roi_label_bls[i] = roi_name
    ax.set_xticklabel_bls(roi_label_bls, rotation=90, ha='right')
    # plt.xticks(np.arange(len(roi_label_bls)), roi_label_bls)
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name)))
    plt.close('all')
############################################################
############MMSE scatter plot
def vis20(comp=0, var = 'brain'):
    id_null = ~np.isnan(continue_pcd[:,2])
    if comp == 0:
        color = 'r'
    elif comp == 1:
        color = '#87CEFA'
    else:
        color = 'grey'
    plt.figure(figsize =(10,10))
    ax = plt.gca()
    plt.scatter(continue_pcd[id_null,2], X_train_[id_null,comp], alpha=.6, color = color)
    name = 'MMSE_correlation_{}_{}'.format(comp, var)
    plt.title(name)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    max_x = continue_pcd[id_null,2].max()
    min_x = continue_pcd[id_null,2].min()
    max_y = X_train_[id_null,comp].max()
    min_y = X_train_[id_null,comp].min()
    r, p = pearsonr(continue_pcd[id_null,2], X_train_[id_null,comp])
    plt.text(abs(max_x - min_x)*0.05+min_x, min_y+abs(max_y - min_y)*0.05, 'R: {:.3}\nP: {:.3}\n'.format(r, p))
    ax.set_xticks(np.around(np.arange(min_x, max_x, abs(max_x - min_x)//3), 1))
    ax.set_yticks(np.around(np.arange(min_y, max_y, abs(max_y - min_y)//3), 1))
    
    ax.set_ylabel_bl('{} {} canonical variates'.format(var, comp+1), fontproperties=font)
    if var == 'brain':
        ax.set_xlabel_bl('MMSE score', fontproperties=font)
    plt.savefig(os.path.join(te_save_path, '{}.svg'.format(name))) 

def correlation_comparison(X_train_, num_cp=2, name = 'Correlates_of_brain_canonical_variates_and_other_measures', y_label_bl=['brain comps 1', 'brain comps 2', 'brain comps 3']):
    id_null = ~np.isnan(continue_pcd)

    r_matrix = np.zeros((num_cp, continue_pcd.shape[-1], 100))
    for i in range(num_cp):
        for j in range(continue_pcd.shape[-1]):
            subjects_oas_ = subjects_oas[id_null[:,j]]
            subjects_uni, subjects_uni_idx, subjects_oas_idx = np.unique(subjects_oas_, return_index=True, return_inverse = True)
            x = X_train_[id_null[:,j],i]
            y = continue_pcd[id_null[:,j],j]
            x_list = []
            y_list = []
            for name_ in subjects_uni:
                x_list.append(np.mean(x[subjects_oas_==name_]))
                y_list.append(np.mean(y[subjects_oas_==name_]))
            
            x_array = np.array(x_list)
            y_array = np.array(y_list)
            for k in range(100):
                idx = np.random.choice(len(x_list), len(x_list))
                x_array_ = x_array[idx]
                y_array_ = y_array[idx]
                r_total_f = pearsonr(x_array_, y_array_)
                r_matrix[i,j,k] = r_total_f[0]
            
    t_p_vector = np.zeros((continue_pcd.shape[-1], 2))
    for j in range(continue_pcd.shape[-1]):
        t_p_vector[j, 0],  t_p_vector[j, 1] = t_test(abs(r_matrix[0,j,:]), abs(r_matrix[1,j,:]), 0.05, True)
    return t_p_vector, abs(r_matrix)
# #######################basic cca result visual
# 'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL'
# t, p = vis4(comp, vari2)
# r = vis1(X_train_)
# p_fdr = multitest.fdrcorrection(np.r_[r[1,:2,0], r[1,:2,1], np.array([0.346, 0.00114,])])[-1]
# p_fdr2 = multitest.fdrcorrection(np.r_[r[1,1,:-4], np.array([0.151, 3.50e-6, 8.46e-4, 5.08e-4, 2.10e-4, 3.65e-6, 0.0011, 0.092, 1.86e-5])])[-1]
# p_fdr2 = multitest.fdrcorrection([0.2498, 6.027e-06, 0.000111, 0.364836, 0.000851, 0.006815, 0.526009, 0.00064, 0.002603, 0.803421, 0.001037, 
#                                   0.0006811, 0.54120, 5.05822e-05, 1.412915e-05, 0.1723396, 0.000755, 0.012028])[-1]
# f,p = vis2(0, vari2)
# p_fdr = multitest.fdrcorrection([3.79e-6, 0.0494, 0.00647, 0.269, 0.234, 0.152, 0.0581, 0.000743, 0.004, 0.0402])[-1]

# vis3(comp, vari2)
# vis5()
comp = 0
# fc_map = vis6(comp, 'roi', 'full', decomp = decomp, roi_num = roi_num, fc_reduce_method2 = fc_reduce_method2)
# sio.savemat(os.path.join(te_save_path, 'weight_comp_{}_{}.mat'.format(comp, vis)), {'weights': fc_map})
# vis7()
# vis8(comp)
# r, p = pearsonr(trans_x_te[:,comp], trans_y_te[:,comp])
# vis9(comp, trans_x_te, trans_y_te, r, p, 'te_cca_comp{}_xy'.format(comp))
# r, p = pearsonr(X_train_[:,comp], y_train_[:,comp])
# vis9(comp, X_train_, y_train_, r, p, 'train_cca_comp{}_xy'.format(comp))
# comp = 0
fc_map = vis10(comp, vis, decomp = decomp,  roi_num = roi_num, fc_reduce_method2 = fc_reduce_method2)
# sio.savemat(os.path.join(te_save_path, 'corr_comp_{}_{}.mat'.format(comp, vis)), {'corr': fc_map})
# HC_mask = label == 'Cognitively normal'
# AD_mask = (label != 'Cognitively normal') & (label != 'uncertain dementia')
# label[AD_mask] = 'Dementia'
# sio.savemat(os.path.join(te_save_path, 'label_diagnosis.mat'), {'label': label})
# sio.savemat(os.path.join(te_save_path, 'subject_name_ab.mat'), {'subjects': subjects_oas.values, 
#                                                               'Centil_fBP_TOT_CORTMEAN': pup_oas_BP_SUVR_list[:,0],
#                                                               'Centil_fSUVR_TOT_CORTMEAN': pup_oas_BP_SUVR_list[:,1]})
# sio.savemat(os.path.join(te_save_path, 'subject_other_measure.mat'), {'subjects': subjects_oas.values, 'CVHATT': heal_his_list[:,0], 
#                   'CVAFIB': heal_his_list[:,1], 'CVANGIO': heal_his_list[:,2], 'CVBYPASS': heal_his_list[:,3], 
#                   'CVPACE': heal_his_list[:,4], 'CVCHF': heal_his_list[:,5], 'CVOTHR': heal_his_list[:,6], 'CBSTROKE': heal_his_list[:,7], 
#                   'CBTIA': heal_his_list[:,8], 'CBOTHR': heal_his_list[:,9], 'PD': heal_his_list[:,10], 'PDOTHR': heal_his_list[:,11],	
#                   'SEIZURES': heal_his_list[:,12], 'TRAUMBRF': heal_his_list[:,13], 'TRAUMEXT': heal_his_list[:,14], 'TRAUMCHR': heal_his_list[:,15],
#                   'NCOTHR': heal_his_list[:,16], 'HYPERTEN': heal_his_list[:,17], 'HYPERCHO': heal_his_list[:,18], 'DIABETES': heal_his_list[:,19],
#                   'B12DEF': heal_his_list[:,20], 'THYROID': heal_his_list[:,21], 'INCONTU': heal_his_list[:,22], 'INCONTF': heal_his_list[:,23],
#                   'DEP2YRS': heal_his_list[:,24], 'DEPOTHR': heal_his_list[:,25], 'ALCOHOL': heal_his_list[:,26], 'TOBAC30': heal_his_list[:,27], 
#                   'TOBAC100': heal_his_list[:,28], 'ABUSOTHR': heal_his_list[:,29], 'PSYCDIS': heal_his_list[:,30], 'ABRUPT': cvd_list[:,0], 
#                   'STEPWISE': cvd_list[:,1], 'SOMATIC': cvd_list[:,2], 'EMOT': cvd_list[:,3], 'HXHYPER': cvd_list[:,4], 'HXSTROKE': cvd_list[:,5], 
#                   'FOCLSYM': cvd_list[:,6], 'FOCLSIGN': cvd_list[:,7], 'HACHIN': cvd_list[:,8], 'CVDCOG': cvd_list[:,9], 'STROKCOG': cvd_list[:,10], 
#                   'CVDIMAG': cvd_list[:,11], 'PDNORMAL': updrs_list[:,0], 'SPEECH': updrs_list[:,1], 'FACEXP': updrs_list[:,2], 
#                   'TRESTFAC': updrs_list[:,3], 'TRESTRHD': updrs_list[:,4], 'TRESTLHD': updrs_list[:,5], 'TRESTRFT': updrs_list[:,6],
#                   'TRESTLFT': updrs_list[:,7], 'TRACTRHD': updrs_list[:,8], 'TRACTLHD': updrs_list[:,9], 'RIGDNECK': updrs_list[:,10], 
#                   'RIGDUPRT': updrs_list[:,11], 'RIGDUPLF': updrs_list[:,12], 'RIGDLORT': updrs_list[:,13], 'RIGDLOLF': updrs_list[:,14], 
#                   'TAPSRT': updrs_list[:,15], 'TAPSLF': updrs_list[:,16], 'HANDMOVR': updrs_list[:,17], 'HANDMOVL': updrs_list[:,18], 
#                   'HANDALTR': updrs_list[:,19], 'HANDALTL': updrs_list[:,20], 'LEGRT': updrs_list[:,21], 'LEGLF': updrs_list[:,22], 
#                   'ARISING': updrs_list[:,23], 'POSTURE': updrs_list[:,24], 'GAIT': updrs_list[:,25], 'POSSTAB': updrs_list[:,26],
#                   'BRADYKIN': updrs_list[:,27], 'GDS': gds_list, 'BILLS': faq_list[:,0], 'TAXES': faq_list[:,1], 'SHOPPING': faq_list[:,2],
#                   'GAMES': faq_list[:,3], 'STOVE': faq_list[:,4], 'MEALPREP': faq_list[:,5], 'EVENTS': faq_list[:,6], 'PAYATTN': faq_list[:,7], 
#                   'REMDATES': faq_list[:,8], 'TRAVEL': faq_list[:,9]})

# # vis11(fc,comp)
# vis12()
# vis13()
# t, p = vis14(comp, vari2)
# print(t)

# vis17(fc_map,comp,vis=vis)
# vis18(fc, 'roi')
# vis19(fc, comp)
# vis20(comp, vari2)

# ###############correlation comparison

# ['apoe', 'mmse', 'DIGIF', 'DIGIB', 'EDU',
#                             'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB', 
#                             'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON', 'sumbox']
# t_p_vector,r_matrix = correlation_comparison(X_train_)

# save_path = r'H:\PHD\learning\research\AD_two_modal\result\final_figure\r_diff'
# name = 'sumbox_difference'
# r1 = r_matrix[0,16,:]
# r2 = r_matrix[1,16,:]
# plt.figure(figsize =(10,10))
# ax = plt.gca()
# sns.kdeplot(r1, shade=True, bw_adjust=.5, alpha=.3, linewidth=3, color = 'salmon', label='variate1')
# # sns.kdeplot(r1, shade=True, bw_adjust=.5, alpha=.3, linewidth=3, color = 'salmon')
# sns.rugplot(r1, alpha=.9, color = 'salmon')
# sns.kdeplot(r2, shade=True, bw_adjust=.5, alpha=.3, linewidth=3, color = '#87CEFA', label='variate2')
# # sns.kdeplot(r2, shade=True, bw_adjust=.5, alpha=.3, linewidth=3, color = '#87CEFA')
# sns.rugplot(r2, alpha=.9, color = '#87CEFA')
# # ax.legend(handles = [a1, a2])
# ax.legend()
# ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
# ax.axes.get_yaxis().set_visible(False)
# # ax.set_yticks(np.arange(0, 50, 1.5))
# ax.set_xticks(np.arange(-0.2, 0.7, .2))
# plt.savefig(os.path.join(save_path, '{}.svg'.format(name)))

# #################################measurement correlation
# r_matrix = np.zeros((continue_pcd.shape[-1]+1, continue_pcd.shape[-1]+1))
# p_matrix = np.zeros((continue_pcd.shape[-1]+1, continue_pcd.shape[-1]+1))

# for i in range(continue_pcd.shape[-1]+1):
#     for j in range(continue_pcd.shape[-1]+1):
#         if i == continue_pcd.shape[-1] and j != continue_pcd.shape[-1]:
#             variate2 = continue_pcd[:,j]
#             null_mask = np.isnan(variate2)
#             variate2 = variate2[~null_mask]
#             variate1 = decret_pcd[:,0][~null_mask]
#         elif j == continue_pcd.shape[-1] and i != continue_pcd.shape[-1]:
#             variate2 = continue_pcd[:,i]
#             null_mask = np.isnan(variate2)
#             variate2 = variate2[~null_mask]
#             variate1 = decret_pcd[:,0][~null_mask]
#         elif j == continue_pcd.shape[-1] and i == continue_pcd.shape[-1]:
#             variate2 = decret_pcd[:,0]
#             variate1 = decret_pcd[:,0]
#         else:
#             variate2 = continue_pcd[:,j]
#             variate1 = continue_pcd[:,i]
#             null_mask2 = np.isnan(variate2)
#             null_mask1 = np.isnan(variate1)
#             null_mask = null_mask1 | null_mask2
#             variate2 = variate2[~null_mask]
#             variate1 = variate1[~null_mask]
#         r, p = pearsonr(variate2, variate1)
#         r_matrix[i,j] = r
#         p_matrix[i,j] = p

# def get_top(r_matrix, p_matrix):
#     top = (p_matrix<=0.05) & (abs(r_matrix)>=0.3)
#     x,y = np.meshgrid(np.arange(r_matrix.shape[1]),np.arange(r_matrix.shape[0]))
#     m = np.c_[x[top],y[top]]
#     for pos in m:
#         rect(pos)
#     return 
# def rect(pos):
#     r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor="black", linewidth=1.5)
#     plt.gca().add_patch(r)
        
# np.fill_diagonal(r_matrix, 0)
# np.fill_diagonal(p_matrix, 1)
# v_min = r_matrix.min()
# v_max = r_matrix.max()
# norm = mcolors.TwoSlopeNorm(vmin=v_min, vmax = v_max, vcenter=0)
# plt.figure(figsize =(15,15))
# ax = plt.gca()
# im = ax.imshow(r_matrix, vmin=v_min, vmax=v_max, cmap = 'RdBu_r', norm=norm)
# cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
# cbar.ax.set_ylabel('R', rotation=-90, va="bottom", fontproperties=font)
# axis_label = np.arange(continue_pcd.shape[-1]+1)
# ax.set_xticks(axis_label-0.1) 
# ax.set_xticklabels(['apoe', 'mmse', 'DIGIF', 'DIGIB', 'EDU',
#                            'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB', 
#                            'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON', 'cdr'], fontproperties = font)
# plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
#           rotation_mode="anchor")
# axis_label = np.arange(continue_pcd.shape[-1]+1)
# ax.set_yticks(axis_label) 
# ax.set_yticklabels(['apoe', 'mmse', 'DIGIF', 'DIGIB', 'EDU',
#                            'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB', 
#                            'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON', 'cdr'], fontproperties=font)
# ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# get_top(r_matrix, p_matrix)
# save_path = r'H:\PHD\learning\research\AD_two_modal\result\final_figure'
# plt.savefig(save_path + '\measure_pearson.svg', bbox_inches = 'tight')

# half_fc_map = np.zeros((fc_map.shape[1] * (fc_map.shape[1]-1)//2))
# idx_ = np.triu_indices_from(fc_map, 1)  
# half_fc_map = fc_map[idx_]
# fc_ = fc[idx,:]
# mask_ = half_fc_map!=0
# half_fc_comp1 = half_feature[:,mask_]
# from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
# # kmeans = KMeans(n_clusters=4, random_state=9).fit(half_feature)
# kmeans = SpectralClustering(n_clusters=4, assign_labels='discretize').fit(half_feature)
# sub_id = kmeans.labels_
# comp1_npi = npi_raw2[sub_id==0]
# comp2_npi = npi_raw2[sub_id==1]
# pearsonr(npi_raw2[sub_id==3][:,3], npi_raw2[sub_id==3][:,4])

# ####################################brain pattern explain
# file_comp1 = r'E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\corr_comp_1_fdr.mat'
# # file_comp2 = r'E:\PHD\learning\research\AD_two_modal\result\multi_run\advanced_analysis\100roi\baseline\cca_12_domains_OAS_norm_True_subsample_zeroTrue_l10.6_pca_npi7_method_none_fmri4950_CAcomp4_fold10\two_component_overlapping.mat'
# corr1 = sio.loadmat(file_comp1)['corr']
# # corr2 = sio.loadmat(file_comp2)['overlap']
# comp1_mask = np.expand_dims(corr1!=0, 0)
# comp1_mask_pos = np.expand_dims(corr1>0, 0)
# comp1_mask_neg = np.expand_dims(corr1<0, 0)
# comp1_idx_pos = keep_triangle_half(comp1_mask_pos.shape[1] * (comp1_mask_pos.shape[1]-1)//2, comp1_mask_pos.shape[0], comp1_mask_pos)
# comp1_idx_neg = keep_triangle_half(comp1_mask_neg.shape[1] * (comp1_mask_neg.shape[1]-1)//2, comp1_mask_neg.shape[0], comp1_mask_neg)
# comp1_idx_all = keep_triangle_half(comp1_mask.shape[1] * (comp1_mask.shape[1]-1)//2, comp1_mask.shape[0], comp1_mask)
# variate1_pos = np.dot(X_train * comp1_idx_pos, w_x)[:,1]
# variate1_neg = np.dot(X_train * comp1_idx_neg, w_x)[:,1]
# variate1_all = np.dot(X_train * comp1_idx_all, w_x)[:,1]
# comp2_mask = np.expand_dims(corr2!=0, 0)
# mask = comp1_mask | comp2_mask
# comp_idx = keep_triangle_half(mask.shape[1] * (mask.shape[1]-1)//2, mask.shape[0], mask)
# variate2_all = np.dot(X_train * comp_idx, w_x)[:,1]

# comp2_mask_pos = np.expand_dims(corr2>0, 0)
# comp2_mask_neg = np.expand_dims(corr2<0, 0)
# comp2_idx_pos = keep_triangle_half(comp2_mask_pos.shape[1] * (comp2_mask_pos.shape[1]-1)//2, comp2_mask_pos.shape[0], comp2_mask_pos)
# comp2_idx_neg = keep_triangle_half(comp2_mask_neg.shape[1] * (comp2_mask_neg.shape[1]-1)//2, comp2_mask_neg.shape[0], comp2_mask_neg)
# variate2_pos = np.dot(X_train * comp2_idx_pos, w_x)[:,1]
# variate2_neg = np.dot(X_train * comp2_idx_neg, w_x)[:,1]
# variate = np.dot(X_train, w_x)
# variate = np.dot(X_train, w_x)
# p = vis3(0, np.expand_dims(variate1_all, 1), 'Brain_cp1_all_NPI_correlation_spear_f')
# p_fdr = multitest.fdrcorrection(p.squeeze())[-1]

# vis1(np.expand_dims(variate1_neg, -1), num_cp=1, name = 'Correlates_brain_cp2_neg_and_other_measures')
# # vis1(np.expand_dims(variate1_all, -1), num_cp=1, name = 'Correlates_of_brain_overlap2_specific_all6_and_other_measures')
#########################################################
# r_matrix1 = np.zeros((half_feature.shape[-1],2))
# r_matrix2 = np.zeros((half_feature.shape[-1],2))
# for i in range(half_feature.shape[-1]):
#     fc = half_feature[:,i]
#     out1 = pearsonr(fc, npi_raw2[:,3])
#     out2 = pearsonr(fc, npi_raw2[:,4])
#     if out1[0] * out2[0] < 0 and (out1[1]<0.05 or out2[1]<0.05):# and abs(out1[0] - out2[0])>0.2:
#         r_matrix1[i,0] = out1[0]
#         r_matrix1[i,1] = out1[1]     
#         r_matrix2[i,0] = out2[0]
#         r_matrix2[i,1] = out2[1]
# p = multitest.fdrcorrection(r_matrix1[:,-1])
# p2 = multitest.fdrcorrection(r_matrix2[:,-1])
# r_matrix1 = r_matrix1[:,0] * (p[0] * 1)
# r_matrix2 = r_matrix2[:,0] * (p2[0] * 1)
# fc_pca_pls_loading_sys, _ = vector_to_matrix(abs(r_matrix2 - r_matrix1))
# # site1_coeff, site1_p_value, _ = mantel(abs(fc_map), fc_pca_pls_loading_sys)

# out_noise = np.random.random((1000,2))
# for i in range(1000):
#     a = np.random.random((100,100))
#     a = a + a.T
#     np.fill_diagonal(a, 0)
#     b = np.random.random((100,100))
#     b = b + b.T
#     np.fill_diagonal(b, 0)
#     out_noise[i,0], out_noise[i,1], _ = mantel(a, b)
#     print(i)
# p = (sum(out_noise[:,0]>0.334)+1)/1001

# fig, ax = plt.subplots(figsize=(15, 15))
# roi_idx = np.arange(0,100,1)
# im, cbar = heatmap(fc_pca_pls_loading_sys, roi_idx, ax=ax, cmap="Reds", 
#                     cbarlabel="Correlation coefficient (abs)",) 
# plt.savefig(os.path.join(te_save_path, '{}.svg'.format('drpession_anxiety_reverse')), bbox_inches = 'tight')

#Depression, Anxiety
# vis15(npi_raw2, 0, 'npi', domain = 'Anxiety')
# 'Age', 'apoe', 'mmse', 'DIGIF', 'DIGIB', 'EDU',
# 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB', 
# 'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON'
# vis16(npi_raw2, comp=15, variable = 'MEMTIME', domain = 'Depression')
# id_null = ~np.isnan(continue_pcd[:,-2])
# a = npi_raw2[id_null,4]
# b = continue_pcd[id_null,-2]
# mask_ = a!=3
# pearsonr(a[mask_], b[mask_])

# plt.figure(figsize =(15,15))
# ax = plt.gca()
# plt.scatter(y_train_[:,0], npi_raw2[:,3], alpha=.6, color = 'r')
# # plt.figure(figsize =(15,15))
# # ax = plt.gca()
# plt.scatter(y_train_[:,0], npi_raw2[:,4], alpha=.6, color = 'b')
# plt.figure(figsize =(15,15))
# ax = plt.gca()
# plt.scatter(npi_raw2[:,3], npi_raw2[:,4], alpha=.6, color = 'g')

# pearsonr(X_train_[:,0], continue_pcd[:,5])
# a = np.zeros((4,3))
# a[0,0] = sum((npi_raw2[:,4]==3) & (npi_raw2[:,3]==0))
# a[1,0] = sum((npi_raw2[:,4]==2) & (npi_raw2[:,3]==0))
# a[2,0] = sum((npi_raw2[:,4]==1) & (npi_raw2[:,3]==0))
# a[3,0] = sum((npi_raw2[:,4]==0) & (npi_raw2[:,3]==0))
# a[0,1] = sum((npi_raw2[:,4]==3) & (npi_raw2[:,3]==1))
# a[1,1] = sum((npi_raw2[:,4]==2) & (npi_raw2[:,3]==1))
# a[2,1] = sum((npi_raw2[:,4]==1) & (npi_raw2[:,3]==1))
# a[3,1] = sum((npi_raw2[:,4]==0) & (npi_raw2[:,3]==1))
# a[0,2] = sum((npi_raw2[:,4]==3) & (npi_raw2[:,3]==2))
# a[1,2] = sum((npi_raw2[:,4]==2) & (npi_raw2[:,3]==2))
# a[2,2] = sum((npi_raw2[:,4]==1) & (npi_raw2[:,3]==2))
# a[3,2] = sum((npi_raw2[:,4]==0) & (npi_raw2[:,3]==2))

# id_null = ~np.isnan(continue_pcd)
# r_matrix = np.zeros((2, 2, continue_pcd.shape[-1]+2))
# p_matrix = np.zeros((2, 2, continue_pcd.shape[-1]+3))
# for i in range(2):
#     for j in range(continue_pcd.shape[-1]):
#         r_total_f = pearsonr(X_train_[id_null[:,j],i], continue_pcd[id_null[:,j],j])
#         r_total_f2 = pearsonr(y_train_[id_null[:,j],i], continue_pcd[id_null[:,j],j])
#         r_matrix[0,i,j] = r_total_f[0]
#         r_matrix[1,i,j] = r_total_f2[0]  
#         p_matrix[0,i,j] = r_total_f[1]
#         p_matrix[1,i,j] = r_total_f2[1]
        
# fdr_correct = np.zeros((2, continue_pcd.shape[-1]))
# # for i in range(2):
# r_matrix[0,0,-2], p_matrix[0,0,-3] = vis2(0, 'brain')
# r_matrix[1,0,-2], p_matrix[1,0,-3] = vis2(0, 'npi')
# r_matrix[0,1,-2], p_matrix[0,1,-3] = vis2(1, 'brain')
# r_matrix[1,1,-2], p_matrix[1,1,-3] = vis2(1, 'npi')
# r_matrix[0,0,-1], p_matrix[0,0,-2] = vis4(0, 'brain')
# r_matrix[1,0,-1], p_matrix[1,0,-2] = vis4(0, 'npi')
# r_matrix[0,1,-1], p_matrix[0,1,-2] = vis4(1, 'brain')
# r_matrix[1,1,-1], p_matrix[1,1,-2] = vis4(1, 'npi')
# p_matrix[0,0,-1] = 0.141
# p_matrix[1,0,-1] = 0.0669
# p_matrix[0,1,-1] = 3.09e-12
# p_matrix[1,1,-1] = 3.34e-20
# out = multitest.fdrcorrection(np.concatenate((p_matrix[0][0], p_matrix[0][1], p_matrix[1][0], p_matrix[1][1])))[-1]
# out = multitest.fdrcorrection([0.007, 0.006, 0.974,0.441,0.262,0.086,0.500])[-1]
# p_out = out.reshape(2, 2, continue_pcd.shape[-1]+3)

# sio.savemat(te_save_path + r'\brain_transformed2.mat', {'brain_feature': X_train_, 'w_x': w_x})
# sio.savemat(te_save_path + r'\npi_transformed2.mat', {'npi_feature': y_train_, 'w_y': w_y, 'weight_npi_fa': npi_reduce_model.components_.T})
# data.to_csv(r'E:\PHD\learning\research\AD_two_modal\result\advanced_analysis\cca_pca4_0.6l1_60fmri\pcd_interest.csv')
######################################nps correlation
# corr = np.corrcoef(npi.T)
# np.fill_diagonal(corr, 0)

# plt.figure(figsize =(15,15))
# ax = plt.gca()
# try:
#     norm = mcolors.TwoSlopeNorm(vmin=corr.min(), vmax = corr.max(), vcenter=0)
#     im = plt.imshow(corr, cmap = 'RdBu_r', norm=norm)
# except ValueError:
#     im = plt.imshow(corr, cmap = 'RdBu_r')
# im_ratio = corr.shape[0]/corr.shape[1] 
# cbar = plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
# cbar.ax.set_ylabel('R', rotation=0, va="bottom", fontproperties=font2)
# # plt.colorbar()
# name = 'npi_corr'
# plt.title(name)
# ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
# ax.spines['bottom'].set_color('none')  # 设置上‘脊梁’为无色
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# ax.axes.get_xaxis().set_visible(False) 
# ax.axes.get_yaxis().set_visible(False) 
# for i in range(corr.shape[0]):
#     for j in range(corr.shape[1]):
#         r,p = pearsonr(npi[:,i], npi[:,j])
#         if p<= 0.05:
#             text = ax.text(j,i,corr[i,j].round(2),ha="center", va="center", color="black",fontsize=20,fontname='Times New Roman')
# # plt.savefig(os.path.join(r'H:\PHD\learning\research\AD_two_modal2\result\final_figure', '{}.svg'.format(name)), bbox_inches = 'tight') 
