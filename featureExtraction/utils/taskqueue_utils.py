from featureExtraction.utils import pss_extraction_utils_updated 
import math 
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink 
import importlib 
import os,sys,inspect 
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
currentdir = "/usr/local/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
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
import cloudvolume
 
from featureExtraction import provider
import importlib

import pandas as pd
from sklearn.cluster import KMeans

import meshparty
import time
from meshparty import trimesh_io
import trimesh
from trimesh.primitives import Sphere
import os
import h5py
from meshparty import skeleton, skeleton_io
import json
import math
import cgal_functions_Module as cfm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from meshparty import mesh_filters
from meshparty import utils
import numpy as np
import pickle
import pandas
import cloudvolume
from meshparty import skeletonize
import pyembree
from trimesh.ray import ray_pyembree
from multiprocessing import Pool
from functools import partial
from scipy import sparse
from meshparty import skeleton_io
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from annotationframeworkclient import infoservice,FrameworkClient
from taskqueue import TaskQueue
from caveclient import CAVEclient

def create_obj(config_file):
    with open(config_file) as f:
      procObj = json.load(f)

    #framework client for database and cv connection for mesh downloads
    client = CAVEclient(procObj['dataset_name'],auth_token_file=procObj['auth_token_file'])
    tokenfile = procObj['auth_token_file']
    with open(tokenfile) as f:
      tokencfg = json.load(f)

    procObj['token'] = tokencfg['token']
    procObj['segsource'] = client.info.segmentation_source()
    
    return procObj

def create_feature_obj(config_file,cell_id):
    
    cfg, data_synapses = create_proc_obj(config_file,cell_id)
    #tensorflow variables
    cfg['tensorflow_model'] = importlib.import_module('models.model') # import network module
    if not os.path.exists(cfg['pointnet_dump_dir']): os.mkdir(cfg['pointnet_dump_dir'])
    cfg['LOG_FOUT'] = open(os.path.join(cfg['pointnet_dump_dir'], 'log_evaluate.txt'), 'w')
    return cfg, data_synapses

def create_proc_obj (config_file,cell_id):
    print("This is cell id: ", cell_id)
       
    with open(config_file) as f:
      procObj = json.load(f)

    #framework client for database and cv connection for mesh downloads
    print ("setting up cave client", procObj['auth_token_file'])
    client = CAVEclient(procObj['dataset_name'],auth_token_file=procObj['auth_token_file'])
    tokenfile = procObj['auth_token_file']
    with open(tokenfile) as f:
      tokencfg = json.load(f)
    print("loaded token")
    procObj['token'] = tokencfg['token']
    procObj['segsource'] = client.info.segmentation_source()
    
    #inits - just cell id and directories
    procObj['cell_id'] = int(cell_id)
    procObj['myfiles'] =  glob.glob(procObj['pss_directory']+'%s/PSS*.txt'%cell_id)
    procObj['offfiles'] =  glob.glob(procObj['pss_directory']+'%s/PSS*.off'%cell_id)
    procObj['outdir'] = procObj['pss_directory']+'/' + procObj["type_of_shape"]+'/%s'%str(cell_id)
    
    if not os.path.exists(procObj['outdir']):
        os.makedirs(procObj['outdir'])

    procObj['d_mesh'] = None
    procObj['sk'] = None
    print("About do do postsynaptic")
    print("This is type of shape: ", procObj["type_of_shape"])

    if procObj["type_of_shape"] == "postsynaptic":
        data_synapses= client.materialize.query_table('synapses_pni_2',filter_in_dict={'post_pt_root_id':['%d'%cell_id]}, materialization_version = procObj['materialization_version'])
        procObj['cell_center_of_mass'] =np.array(client.materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cell_id]}, materialization_version = procObj['materialization_version'])['pt_position'].values[0])
        
        
    elif procObj["type_of_shape"] == "presynaptic":
        data_synapses= client.materialize.query_table('synapses_pni_2',filter_in_dict={'pre_pt_root_id':['%d'%cell_id]}, materialization_version = procObj['materialization_version'])
        procObj['cell_center_of_mass'] = None
    else: 
        print("Did not recognize shape type! Should be one of : presynaptic, postsynaptic")
    print("Done with that") 
    print (cell_id)
    procObj['nucleus_id'] =np.array(client.materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cell_id]}, materialization_version = procObj['materialization_version'])['id'].values[0])
    print("Next one")

    
    
    #procObj['synapse_ids'] = data_synapses.id
    
    # which synapses to process
    
    #if procObj['startPSS'] > -1:
    #   startpss = procObj['startPSS']
    #else:
    #    startpss = 0
    
    #if procObj['stopPSS'] > -1:
    #    stoppss = procObj['stopPSS']
    #else:
    #    stoppss = len(data_synapses)
    
    
    #if procObj['selectedPSS'] < 0:
    #    procObj['rng'] =  range(startpss,stoppss)
    
    #else:
    #    procObj['rng'] = range(procObj['selectedPSS'],procObj['selectedPSS']+1)
        
    return procObj,data_synapses
