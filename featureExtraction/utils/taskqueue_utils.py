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
from taskqueue import TaskQueue



def create_proc_obj (config_file,cell_id):
       
    with open(config_file) as f:
      procObj = json.load(f)

    #framework client for database and cv connection for mesh downloads
    procObj['client'] = FrameworkClient(cfg['dataset_name'],auth_token_file=cfg['auth_token_file'])
    tokenfile = procObj['auth_token_file']
    with open(tokenfile) as f:
      tokencfg = json.load(f)

    procObj['token'] = tokencfg['token']
    procObj['segsource'] = procObj['client'].info.segmentation_source()
    
    #inits - just cell id and directories
    procObj['cell_id'] = int(cell_id)
    procObj['myfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.txt'%cell_id)
    procObj['offfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.off'%cell_id)
    procObj['outdir'] = cfg['pss_directory']+'/' + cfg["type_of_shape"]+'/%s'%str(cell_id)
    
    if not os.path.exists(procObj['outdir']):
        os.makedirs(procObj['outdir'])

    procObj['d_mesh'] = None
    procObj['sk'] = None
    
    if procObj["type_of_shape"] == "postsynaptic":
        procObj['data_synapses']= cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'post_pt_root_id':['%d'%cell_id]})
        procObj['cell_center_of_mass'] =cfg['client'].materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cell_id]})['pt_position'].values[0]
        
    elif procObj["type_of_shape"] == "presynaptic":
        procObj['data_synapses']= cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'pre_pt_root_id':['%d'%cell_id]})
        procObj['cell_center_of_mass'] = None
    else: 
        print("Did not recognize shape type! Should be one of : presynaptic, postsynaptic")
      
    print("This is the length of synapses: ", procObj['data_synapses'].shape)
    
    
    # which synapses to process
    
    if procObj['startPSS'] > -1:
        startpss = procObj['startPSS']
    else:
        startpss = 0
    
    if procObj['stopPSS'] > -1:
        stoppss = procObj['stopPSS']
    else:
        stoppss = len(procObj['data_synapses'])
    
    
    if procObj['selectedPSS'] < 0:
        procObj['rng'] =  range(startpss,stoppss)
    
    else:
        procObj['rng'] = range(procObj['selectedPSS'],procObj['selectedPSS']+1)
        
    return procObj