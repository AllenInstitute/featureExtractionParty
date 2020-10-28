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
print("Finished imports")

#Read all the Synapses in the database
def read_synapses(cell_id):
    data_synapses = dl.query_synapses('pni_synapses_i1', post_ids = [cell_id])
    #data_synapses = data[data['postsyn_segid']==cell_id]
    return data_synapses

#READ ALL the 1024 feature files in it
def read_1024features(cell_id):
    file_directory = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/%s'%cell_id
    feature_files  = glob.glob(file_directory+'/*manualV3.txt')
    return feature_files

#NOTE: The number of synapses might be more than the number of .txt files. 

#Apply Victoria's classifier
def classify(cell_id,data_synapses):
    # Import classifier model (SVC with linear kernel)
    
    print("This is the first data synapse:")
    print(data_synapses.iloc[0])
    
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
            
            if feature_files[j].find("PSS_"+str(synapseid)+"_"+str(i)+"_ae") != -1:
                exists = True
        if exists == True:
            #features_file = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/' + str(cell_id) + '/spine_' + str(i) +'_ae_model_manualV3.txt'
            features_file = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/' + str(cell_id) + '/PSS_' + str(synapseid) + "_" + str(i) +'_ae_model_manualV3.txt'
            mesh_file = features_file.replace('_ae_model_manualV3.txt','.off')
            
            
            features = np.loadtxt(features_file)
            features = features.transpose()
            
            feat_emb.append(features)
            allfullfeaturesfiles.append(mesh_file)
            good_inds.append(i)
        else:
            features = np.ones(1024)*-100
            feat_emb.append(features)
            allfullfeaturesfiles.append('')
            
            
            
    newfeatemb = np.array(feat_emb)
    umap2d = reducer.transform(newfeatemb)
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
def populate_dataframe():
    #data_synapses['class (linear SVC, 1024 features)'] = pred
    #data_synapses['decision function'] = dec_func
    #print(data_synapses.shape)
    #print(feature_embedding[0,:].shape)
    print("This is hte shape of features")
    print(len(features), len(data_synapses))
    data_synapses['umap0'] = feature_embedding0
    data_synapses['umap1'] = feature_embedding1
    data_synapses['feature1024'] = features
    data_synapses['pss_mesh_file'] = myfiles
    return data_synapses

# Save to output file
def save_output():
    outputfile = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/basil/PSS/classification/synapses/PSS_UMAP_%d.pkl'%cell_id
    pickle.dump(data_synapses_full,open(outputfile, "wb" ))
    
