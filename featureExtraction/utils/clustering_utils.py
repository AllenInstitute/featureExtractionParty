import umap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster, spatial, stats
#from dynamicTreeCut import cutreeHybrid, dynamicTreeCut
import seaborn as sns
import networkx as nx
import phenograph
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_predict
import os
import json
from scipy.stats import wasserstein_distance
from pyemd import emd, emd_with_flow
import os
import dash
import pandas as pd
from dashdataframe import configure_app
from nglui.statebuilder import *
import annotationframeworkclient
from annotationframeworkclient import imagery
import random

#Finished


def distmatfunc(i,j,numshollbins):
    
    DM_E = myobj['Euclidean']
    DM_D = myobj['Depth_Matrix']
    DM_W = myobj['EarthMovers_%dbins'%numshollbins]
    
    i = int(i[0]) 
    j = int(j[0]) 
    return facE*DM_E[i,j] + facD*DM_D[i,j] + facW*DM_W[i,j]


def create_random_colors(numcolors):
    allcolors = []
    for i in range(0, numcolors):
        clr = [ round(random.uniform(0, 1),1),round(random.uniform(0, 1),1),round(random.uniform(0, 1),1)]
        allcolors.append(clr)
        
        
    return allcolors


#functions
def calcdistance(x,y):
    #print("This is length of x and y", len(x), " ", len(y))
    inds_was = list(np.where(emd_vec==1)[0])
    inds_euc = list(np.where(emd_vec==0)[0])
    inds_depth = list(np.where(emd_vec==2)[0])
    
    wasx = [x[i] for i in inds_was ]
    wasy = [y[i] for i in inds_was ]
    eucx = [x[i] for i in inds_euc ]
    eucy = [y[i] for i in inds_euc ]
    depx = [x[i] for i in inds_depth ]
    depy = [y[i] for i in inds_depth ]
    
    d1 = np.linalg.norm(np.array(eucx)-np.array(eucy))
    d3 = np.linalg.norm(np.array(depx)-np.array(depy))
    
    #d2 = wasserstein_distance(wasx,wasy)
    d2 = 0
    numbins = int(len(wasx)/30)
    
    
    for i in range(0,numbins): # distances
        wx = [wasx[j] for j in range(i,len(wasx),30)]
        wy = [wasy[j] for j in range(i,len(wasy),30)]    
        ret = emd(np.array(wx),np.array(wy),distmat1,extra_mass_penalty=-1.0)
        d2+= ret
        
    
    return d1,d2,d3
    

def get_dataframe(cfg):
    surfaceareadict = {}
    for i in range(0,cfg['number_of_total_sholl_bins']):
        key = 'surfacearea_%d'%i
        value = 'surfacearea_%d_to_%d'%(i*cfg['binsize'], (i+1)*cfg['binsize'])
        surfaceareadict[key] = value

    soma_df = joblib.load(open(cfg['filename_soma'], 'rb'))
    surfacearea_df = joblib.load(open(cfg['filename_surfacearea'], 'rb'))
    surfacearea_df = surfacearea_df.rename(columns=surfaceareadict)
    pss_df = joblib.load(open(cfg['filename_pss'], 'rb'))
    pss_df = pss_df.rename(columns={"id": "soma_id"})

    merge1 = soma_df.merge(surfacearea_df,on='soma_id')
    df = merge1.merge(pss_df,on='soma_id')

    neuron_Tags = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16]
    #neuron_df = df.loc[ (df['cluster'].isin(neuron_Tags)) & (df['surface_area'].values > -1)]
    neuron_df = df.loc[ (df['cluster'].isin(neuron_Tags)) ]
    return neuron_df

def get_feature_names(globvars,df):
    featurenames = dict()

    featurenames['soma_features'] = ['soma_synapses', 'soma_area', 'soma_area_to_volume',
           'soma_syn_density', 'soma_volume',
           'nucleus_area_largest', 'nucleus_area_to_volume_largest',
           'nucleus_avg_depth_largest',
           'nucleus_fold_area_largest', 'nucleus_fract_fold_largest',
           'nucleus_volume_largest', 'nucleus_to_soma', 'nucleus_to_soma_fixed',
           'nucleus_to_soma_largest','synapse_size_mean_x','synapse_size_median_x']

    featurenames['depth_features'] = ['y']

    maxsholldist = globvars['number_of_sholl_bins']*globvars['binsize']
    pss_features = []
    sa_features = []
    for s in df.keys().values:
        tokens = s.split('_')
        featdist = tokens[-1]
        if ('shape' in s) :
            if (int(featdist) <= int(maxsholldist)):
                pss_features.append(s)
                
        if ('surfacearea_' in s):
            if (int(featdist) <= int(maxsholldist)):
                sa_features.append(s)
    featurenames['pss_features'] = pss_features   
    featurenames['surface_area'] = sa_features
    featurenames['tags'] = ['Tags_LE', 'Tags_FC', 'QClabels']

    return featurenames

def calculate_diffs(neuron_df):
    #pss_features = featurenames['pss_features']
    #for i in range(0,len(pss_features)):
    #    mymet = pss_features[i]

    #    tokens = mymet.split('_')
    #    st,distance = tokens[2].split('-')
    #    sval = 'surfacearea_bin_%s'%distance
        
    #    if int(distance) > 15:
    #        prevdistance = int(distance)-15
    #        myprevmet = mymet.replace(distance,str(prevdistance))
    #        neuron_df.loc[:, mymet] = neuron_df.loc[:, mymet]-neuron_df.loc[:, myprevmet]
    #    else:
    #        a = 1
    
    for i in range(15,1,-1):
        mymet = 'surfacearea_%d_to_%d'%(i*15000, (i+1)*15000)
        prevmet = 'surfacearea_%d_to_%d'%((i-1)*15000, i*15000)
        neuron_df.loc[:, mymet]= neuron_df.loc[:, mymet] - neuron_df.loc[:, prevmet]
            

def normalize_by_surface_area(neuron_df,featurenames):
    pss_features = featurenames['pss_features']
    surface_area_features = featurenames['surface_area']
    for pfeat in pss_features:
        pref,distval = pfeat.split('dist_')
        safeat = 'surfacearea_%s'%distval
        origpfeat = 'orig_'+pfeat
        neuron_df.loc[:,origpfeat] = neuron_df.loc[:,pfeat]
        neuron_df.loc[:,pfeat] = neuron_df.loc[:,pfeat]/(neuron_df.loc[:,safeat]+0.0001)
    return neuron_df
        
def calculate_Matrix(globvars,featurenames, neuron_df):
    soma_features = featurenames['soma_features']
    depth_features = featurenames['depth_features']
    pss_features = featurenames['pss_features']
    
    #normalize pss features
    if globvars['normalize_by_surfacearea']:
        neuron_df = normalize_by_surface_area(neuron_df,featurenames)
       
    #select metric values and create matrix M
    mt = []

    for i in range(0,globvars['soma_weight']):
        mt.extend(soma_features)

    numeuc = len(mt)
    for i in range(0,globvars['depth_weight']):
        mt.extend(depth_features)

    
    for i in range(0,globvars['pss_weight']):
        mt.extend(pss_features)

    numtot = len(mt)

    M1,m = normmat(np.array(neuron_df[soma_features].values, dtype=np.float64), 'zscore')
    M2,m = normmat(np.array(neuron_df[depth_features].values, dtype=np.float64), 'zscore')
    #M3,maxs = normmat(np.array(neuron_df[pss_features].values, dtype=np.float64), 'maxscore')
    M3 = np.array(neuron_df[pss_features].values, dtype=np.float64)
    M3 = M3/np.max(M3)
    M = np.concatenate((M1,M2,M3),axis=1)
    maxs=1
    
    ishist = np.ones(numtot)  #value 1 for histogram vectors
    ishist[0:numeuc] = 0      #Leila's features
    ishist[numeuc] = 2        #depth value
    emd_vec = ishist
    
    return M, emd_vec, M3, maxs

def normmat(M,how):
    if how=='zscore':
        means=np.min(M, axis=0)
        stds=np.std(M, axis=0)
        M = (M - means)/stds 
        maxs=0
    else:
        a = 'do nothing'
        #maxs = np.max(M,axis=0)
        #stds=np.std(M, axis=0)
        #M = M/stds
    return M,stds
    
    
