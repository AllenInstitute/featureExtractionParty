import tensorflow as tf
import numpy as np
import argparse
import socket
import time
import os
import scipy.misc
import sys
import glob
import umap
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold, preprocessing


import os,sys,inspect
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
currentdir = "/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
import provider
import importlib

import pandas as pd
from sklearn.cluster import KMeans

import meshparty
import time
#from meshparty import trimesh_io
import trimesh
from trimesh.primitives import Sphere
import os
import h5py
#from meshparty import skeleton, skeleton_io
import json
import math
import cgal_functions_Module as cfm
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
#from meshparty import mesh_filters
#from meshparty import utils
import numpy as np
import pickle
import pandas
import cloudvolume
#from meshparty import skeletonize
import pyembree
from trimesh.ray import ray_pyembree
from multiprocessing import Pool
from functools import partial
from scipy import sparse
#from meshparty import skeleton_io
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from annotationframeworkclient import infoservice
from itkwidgets import view
import joblib
import sys
sys.setrecursionlimit(10000)
from cloudfiles import CloudFiles
print("Finished imports")


def setup_dictionary(dictionaryfile):
    cluster_centers = pd.read_pickle(dictionaryfile)
    kmeans = KMeans(n_clusters=30, init='k-means++', max_iter=3000, n_init=10, random_state=0)
    kmeans.fit(cluster_centers)
    kmeans.cluster_centers_ = np.array(cluster_centers)
    return kmeans

def generate_shape_histograms(pss,dictionarymodel):
    featureslist = list(pss['feature1024'].values)
    features=np.stack( featureslist, axis=0 )
    labels = dictionarymodel.predict(features)
    score = dictionarymodel.score(features)
    hist,f = np.histogram(labels,bins=30)
    return labels,score,hist

def create_empty_dataframe_dict():
    mydict = {}
    mydict['allids'] = []
    mydict['allhists'] = []
    mydict['allpreds'] = []
    mydict['allscores'] = []
    mydict['allpsssholl'] = []
    return mydict

def save_pss_dataframe(outfilename,dict):
    index = 0
    for shape in range(0,30):
        for distind in range(0,16):
            name = 'shape_%d_dist_%d_to_%d'%(shape,sholl_radii_lower[distind],sholl_radii_upper[distind])
            mylist = np.array(allpsssholl)[:,index]
            dict[name] = mylist
            index+=1
    df = pd.DataFrame(dict)
    df.to_pickle(outfilename, compression='infer')
    
def get_sholl_radii(sholl_params):
    sholl_upper = range(sholl_params[0], sholl_params[1],sholl_params[2])
    sholl_lower = range(sholl_params[0]-sholl_params[2], sholl_params[1]-sholl_params[2],sholl_params[2])
    return sholl_lower,sholl_upper

def generate_distance_bins(neuron_df, data_synapses,cell_id):
    cell_center=neuron_df[neuron_df['soma_id']==cell_id]['nucleus_center_mass_nm'].values[0]
    synapse_pts =data_synapses['ctr_pt_position'].values
    synapse_pts = synapse_pts.transpose()
    synapse_pts = np.stack(synapse_pts)*[4,4,40]
    dists = np.linalg.norm(synapse_pts  - cell_center, axis = 1)
    return dists


def read_1024features(cell_id,cfg):
    file_directory = '%s/%s/%s'% (cfg['pss_directory'] , cfg['type_of_shape'], cell_id)
    
    #file_directory = '%s/%s'%(cfg['pss_directory'],cell_id)
    feature_files  = glob.glob(file_directory+'/*manualV3.txt')
    
    return feature_files

def loadCloudJsonFile (Obj,filenames):
    #read mesh
    cloudbucket = Obj['cloud_bucket']
    with open(Obj['google_secrets_file'], 'r') as file:
        secretstring = file.read().replace('\n', '')
    cf = CloudFiles(cloudbucket,secrets =secretstring)
    f = cf.get_json(filenames)
    return f

def classify_cloud(data_synapses,Obj):
    filenames = []

    for i, row in data_synapses.iterrows():
        filenames.append('%s/features/PSS_%d_ae_model_manualV3.json'%(Obj['type_of_shape'],row.id))

    feat_emb = loadCloudJsonFile(Obj,filenames)
    
    
    badinds = [i for i in range(len(feat_emb)) if feat_emb[i] == None]
    
    
    
    try:
        data_synapses.reset_index(inplace=True)
    except:
        print("data_synapses already reindexed")
    
    data_synapses.drop(data_synapses.index[badinds], inplace=True)
    
    feat_emb = list(filter(None, feat_emb))
    
    
    umap2d = Obj['reducer'].transform(np.array(feat_emb))
    #print(umap2d.shape)
    #umap0 = np.ones(data_synapses.shape[0])*-1000
    #umap1 = np.ones(data_synapses.shape[0])*-1000
    #index = 0
    #for g in range(newfew):
    #    umap0[g] = umap2d[index,0] 
    #    umap1[g] = umap2d[index,1]
    #    index+=1
    umap0 = umap2d[:,0]
    umap1 = umap2d[:,1]
    
    return umap0,umap1,feat_emb,data_synapses



def classify(feature_files,cell_id,data_synapses,cfg):
    # Import classifier model (SVC with linear kernel)
    
    #Create predictions list and decision function list
    pred = []
    dec_func = []
    feat_emb = []
    feat_empty = []
    good_inds = []
    umap0 = []
    umap1 = []
    allfullfeaturesfiles = []
    
    
    for i in range(0,data_synapses.shape[0]):
        exists = False    
        synapseid = data_synapses.iloc[i]['id']
        
        for j in range (len(feature_files)):
            
            if feature_files[j].find("PSS_"+str(synapseid)+"_") != -1:
            #if feature_files[j].find("PSS_"+str(synapseid)+"_"+str(i)+"_ae") != -1:
                exists = True
                features_file = feature_files[j]
                
        if exists == True:
            print("Exists is true")
            #features_file = cfg['pss_directory'] + '/' + cfg['type_of_shape']+ '/' + str(cell_id) + '/PSS_' + str(synapseid) + "_" + str(i) +'_ae_model_manualV3.txt'
            #features_file = feature_files[j]
            mesh_file = features_file.replace('_ae_model_manualV3.txt','.off')
            
            features = np.loadtxt(features_file)
            features = features.transpose()
            
            feat_emb.append(features)
            allfullfeaturesfiles.append(mesh_file)
            print('appending good inds')
            good_inds.append(i)
        else:
            print("BAD INDS")
            features = np.ones(1024)*-100
            feat_emb.append(features)
            allfullfeaturesfiles.append('')
            
            
            
    newfeatemb = np.array(feat_emb)
    umap2d = cfg['reducer'].transform(newfeatemb)
    print(umap2d.shape)
    umap0 = np.ones(data_synapses.shape[0])*-1000
    umap1 = np.ones(data_synapses.shape[0])*-1000
    index = 0
    for g in good_inds:
        umap0[g] = umap2d[index,0] 
        umap1[g] = umap2d[index,1]
        index+=1
    return umap0,umap1,feat_emb,allfullfeaturesfiles

#Populate dataframe
def populate_dataframe(data_synapses,features,feature_embedding0,feature_embedding1,myfiles=None):
    #data_synapses['class (linear SVC, 1024 features)'] = pred
    #data_synapses['decision function'] = dec_func
    #print(data_synapses.shape)
    #print(feature_embedding[0,:].shape)
    #print("This is hte shape of features")
    #print(len(features), len(data_synapses))
    data_synapses['umap0'] = feature_embedding0
    data_synapses['umap1'] = feature_embedding1
    data_synapses['feature1024'] = features
    if myfiles is None:
        print("No files")
    else:
        data_synapses['pss_mesh_file'] = myfiles
    return data_synapses
    