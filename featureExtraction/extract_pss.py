from featureExtraction.utils import pss_extraction_utils_updated
import math
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
import importlib
import os,sys,inspect
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
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


def update_synapses(data_synapses,cell_center_of_mass,threshold):
    #print("starting update synapses")
    #print(np.stack(data_synapses.ctr_pt_position.values).shape)
    #print(cell_center_of_mass.shape)
    dists = np.linalg.norm((np.stack(data_synapses.ctr_pt_position.values) - cell_center_of_mass ) * [4,4,40], axis=1)
    data_synapses['dists'] = dists
    return data_synapses[ data_synapses['dists'] < threshold]
        

def extract_shape_updated(cfg,cell_id,run):
    procObj = {}
    
    #from cfg
    procObj['synapse_scale'] = cfg['synapse_scale']
    procObj['local_dist_thresh'] = cfg['local_dist_thresh']
    procObj['cgal_number_of_rays'] = cfg['cgal_number_of_rays']
    procObj['cgal_number_of_clusters'] = cfg['cgal_number_of_clusters']
    procObj['cgal_smoothing_lambda'] = cfg['cgal_smoothing_lambda']
    procObj['cgal_cone_angle'] = cfg['cgal_cone_angle']
    procObj['pointnet_GPU_INDEX'] = cfg['pointnet_GPU_INDEX']
    procObj['segsource'] = cfg['segsource']
    procObj['token'] = cfg['token']
    procObj['type_of_shape'] = cfg['type_of_shape']
    procObj['cloud_bucket'] = cfg['cloud_bucket']
    
    #inits - just cell id and directories
    procObj['cell_id'] = int(cell_id)
    procObj['myfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.txt'%cell_id)
    procObj['offfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.off'%cell_id)
    procObj['outdir'] = cfg['pss_directory']+'/' + cfg["type_of_shape"]+'/%s'%str(cell_id)
    
    if not os.path.exists(procObj['outdir']):
            os.makedirs(procObj['outdir'])

    procObj['d_mesh'] = None
    procObj['sk'] = None
    print("This is my cell id: ",cell_id)
    if cfg["type_of_shape"] == "postsynaptic":
        #print("Starting postsynaptic")
        procObj['data_synapses']= cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'post_pt_root_id':['%d'%cell_id]}, materialization_version = cfg['materialization_version'])
        #print("next")
        procObj['cell_center_of_mass'] =cfg['client'].materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cell_id]},  materialization_version = cfg['materialization_version'])['pt_position'].values[0]
        #print("starting update")
        procObj['data_synapses'] = update_synapses(procObj['data_synapses'], procObj['cell_center_of_mass'] , cfg['syn_distance_threshold'])
        #print("update done")
    elif cfg["type_of_shape"] == "presynaptic":
        procObj['data_synapses']= cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'pre_pt_root_id':['%d'%cell_id]}, materialization_version = cfg['materialization_version'])
        procObj['cell_center_of_mass'] = None
    else: 
        print("Did not recognize shape type! Should be one of : presynaptic, postsynaptic")
      
    print("This is the length of synapses: ", procObj['data_synapses'].shape)
    
    if cfg['startPSS'] > -1:
        startpss = cfg['startPSS']
    else:
        startpss = 0
    
    if cfg['stopPSS'] > -1:
        stoppss = cfg['stopPSS']
    else:
        stoppss = len(procObj['data_synapses'])
    
    
    if cfg['selectedPSS'] < 0:
        procObj['rng'] =  range(startpss,stoppss)
    
    else:
        procObj['rng'] = range(cfg['selectedPSS'],cfg['selectedPSS']+1)
        
    #print("settingup cfg")
    #print(procObj['rng'])
        
    #if cfg['type_of_shape'] == 'postsynaptic':
    #    procObj['cell_center_of_mass'] =cfg['client'].materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cell_id]})['pt_position'].values[0]
    #else:
    #    procObj['cell_center_of_mass'] = None
        
    #print("done with query")
    
    #print(procObj['cell_center_of_mass'])
    procObj['mesh_bounds'] = cfg['mesh_bounds']
    
    procObj['client'] = cfg['client']
    procObj['forcerun'] = cfg['forcerun']
    #procObj['cv'] = cfg['cv']
    
    #spinemesh= pss_extraction_utils_updated.myprocessingfunc(procObj,0,72)
    #return spinemesh
    #print("Debug 0")
    if run == "parallel" :
        #print("Doing parallel")
        obj = pss_extraction_utils_updated.myParallelProcess(procObj)
    elif run == "parallelTasks":
        #print("Doing parallel tasks")
        obj = pss_extraction_utils_updated.myParallelTasks(procObj)
    elif run == "noextraction":
        print("No Extraction")
        obj = None
    else:
        #print("Doing serial")
        obj = pss_extraction_utils_updated.mySerialProcess(procObj)
        
    #print("Debug 1 : ", type(obj))
    
    return procObj,obj


    
def generate_features(procObj):
    procObj['pointnet_files'] = glob.glob(procObj['outdir']+'/PSS*.off')
    tf.reset_default_graph()
    pss_extraction_utils_updated.evaluate(procObj)
    tf.reset_default_graph()


def extract_shape_and_generate_features_updated(config_file,cell_id_list,run="parallel"):
    with open(config_file) as f:
      cfg = json.load(f)
    
    #framework client for database and cv conneciton for mesh downloads
    cfg['client'] = FrameworkClient(cfg['dataset_name'],auth_token_file=cfg['auth_token_file'])

    
    #seg_source = cfg['client'].info.segmentation_source()
    tokenfile = cfg['auth_token_file']
    with open(tokenfile) as f:
      tokencfg = json.load(f)
    #cfg['cv'] = cloudvolume.CloudVolume(seg_source, use_https=True,secrets=tokencfg['token'])
    
    cfg['token'] = tokencfg['token']
    cfg['segsource'] = cfg['client'].info.segmentation_source()
    
    #tensorflow variables
    cfg['tensorflow_model'] = importlib.import_module('models.model') # import network module
    if not os.path.exists(cfg['pointnet_dump_dir']): os.mkdir(cfg['pointnet_dump_dir'])
    cfg['LOG_FOUT'] = open(os.path.join(cfg['pointnet_dump_dir'], 'log_evaluate.txt'), 'w')
    
    #Processing
    for cell_id in cell_id_list:
        #print(cell_id)
        procObj,obj = extract_shape_updated(cfg,cell_id,run)
        cfg['offfiles'] = procObj['offfiles']
        cfg['outdir'] = procObj['outdir']
        
        procObj = generate_features(cfg)
        
    return procObj, obj
    
    
