from featureExtraction.utils import pss_extraction_utils
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
from annotationframeworkclient import infoservice



def extract_shape(cfg,cell_id):

    procObj = {}
    
    #from cfg
    procObj['local_dist_thresh'] = cfg['local_dist_thresh']
    procObj['cgal_number_of_rays'] = cfg['cgal_number_of_rays']
    procObj['cgal_number_of_clusters'] = cfg['cgal_number_of_clusters']
    procObj['cgal_smoothing_lambda'] = cfg['cgal_smoothing_lambda']
    procObj['cgal_cone_angle'] = cfg['cgal_cone_angle']
    procObj['pointnet_GPU_INDEX'] = cfg['pointnet_GPU_INDEX']
    
    #inits - just cell id and directories
    procObj['cell_id'] = int(cell_id)
    procObj['myfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.txt'%cell_id)
    procObj['offfiles'] =  glob.glob(cfg['pss_directory']+'%s/PSS*.off'%cell_id)
    procObj['outdir'] = cfg['pss_directory']+'/%s'%str(cell_id)
    if not os.path.exists(procObj['outdir']):
            os.makedirs(procObj['outdir'])

    #download mesh,skeleton and synapses and center of mass and set values
    mm=trimesh_io.MeshMeta(cv_path=cfg['cv_path'], disk_cache_path=procObj['outdir'])
    procObj['d_mesh'] = mm.mesh(seg_id=int(cell_id), merge_large_components=False)
    skelfile = procObj['outdir']+'/%s_skeleton.h5'%int(cell_id)
    if not os.path.exists(skelfile):
        procObj['sk'] = skeletonize.skeletonize_mesh(procObj['d_mesh'],verbose=False)
        skeleton_io.write_skeleton_h5(procObj['sk'], skelfile)
    else:
        procObj['sk'] = skeleton_io.read_skeleton_h5(skelfile)
    procObj['data_synapses'] = cfg['dl'].query_synapses('pni_synapses_i1', post_ids = [int(cell_id)])
    cell_center_of_mass_list = cfg['neuron_df'][cfg['neuron_df']['soma_id']==int(cell_id)]['nucleus_center_mass_nm'].values
    procObj['cell_center_of_mass'] = np.array(cell_center_of_mass_list[0])
    procObj['rng'] =  range(0,len(procObj['data_synapses']))
    
    
    #process each pss
    pss_extraction_utils.myParallelProcess(procObj)
    
    #l = len(procObj['rng'])
    #from multiprocessing import Pool
    #from contextlib import closing
    #with closing( Pool(20) ) as p:
    #    partial_process = partial(myprocessingfunc,procObj,l)
    #    rng = procObj['rng']
    #    p.map(partial_process,rng)
    
    return procObj

def imyprocessingfunc (mynewdict,l,m):
    print('m',mynewdict['cell_id'])
    
def generate_features(procObj):
    procObj['pointnet_files'] = glob.glob(procObj['outdir']+'/PSS*.off')
    tf.reset_default_graph()
    pss_extraction_utils.evaluate(procObj)
    tf.reset_default_graph()


def extract_shape_and_generate_features(config_file,cell_id_list = None):
    with open(config_file) as f:
      cfg = json.load(f)
    
    #INITS
    cfg['dl'] = AnalysisDataLink(dataset_name=cfg['dataset_name'],
                         sqlalchemy_database_uri=cfg['sqlalchemy_database_uri'],
                         materialization_version=cfg['data_version'],
                         verbose=False)

    cfg['tensorflow_model'] = importlib.import_module('models.model') # import network module
    if not os.path.exists(cfg['pointnet_dump_dir']): os.mkdir(cfg['pointnet_dump_dir'])
    cfg['LOG_FOUT'] = open(os.path.join(cfg['pointnet_dump_dir'], 'log_evaluate.txt'), 'w')
    
    
    #start processing
    cfg['neuron_df'] = pd.read_pickle(cfg['input_cell_db'])
    if (cell_id_list == None):
        cell_id_list = list(neuron_df['soma_id'])
        
    for cell_id in cell_id_list:

        procObj = extract_shape(cfg,cell_id)
        cfg['offfiles'] = procObj['offfiles']
        cfg['outdir'] = procObj['outdir']
        procObj = generate_features(cfg)


def myprocessingfunc(Obj,l,q):
    print ("%d out of %d "%(q,l))
    
    s = [4*Obj['data_synapses'].iloc[q]['ctr_pt_position'][0], 4*Obj['data_synapses'].iloc[q]['ctr_pt_position'][1], 40*Obj['data_synapses'].iloc[q]['ctr_pt_position'][2]]

    print(s)
    allmeshes, vertlabels,loc_mesh,pt,sdf,seg = pss_extraction_utils.get_segments_for_synapse(Obj,s)
    
    
    time_start = time.time()
    csg = loc_mesh._create_csgraph()
    ccs = sparse.csgraph.connected_components(csg)
    ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
    large_cc_ids = ccs_u[cc_sizes > 20]
    etime = time.time()-time_start
    
    dist_to_center = np.linalg.norm(s-Obj['cell_center_of_mass'])
    
    print("Distance to center: ", dist_to_center, "cellid = ", Obj['cell_id'] )
    
    if (dist_to_center < 15000) & (np.max(cc_sizes) >5000):
        
        spinemesh = pss_extraction_utils.create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
        
    else:
                
        try:
            path = pss_extraction_utils.find_path_skel2synapse_cp(loc_mesh,sk,pt)
        except:
            path = None

        if path is None:
            spinemesh = pss_extraction_utils.create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
        else:
            start_time = time.time()
            pathlabels = pss_extraction_utils.find_mesh_order(path,vertlabels)
            
            if len(pathlabels) > 1: #only cases with more than one segment (those will be either good ones or shafts)
                sdf_verts = pss_extraction_utils.assign_labels_to_verts(loc_mesh,sdf)
                sdf_mean = []
                for ind in range(0,len(pathlabels)):
                    lastmesh = allmeshes[pathlabels[ind]][0]
                    t1 = pss_extraction_utils.get_indices_of_path(loc_mesh, lastmesh, path)
                    sdfvec = [sdf_verts[t] for t in t1]
                    sdf_mean.append(np.mean(sdfvec))

                if sdf_mean[-1] > sdf_mean[-2]:
                    pathlabels = pathlabels[:-1]
            spinemeshes = [allmeshes[p] for p in pathlabels ]
            
            spinemesh = trimesh.util.concatenate(spinemeshes)
            
            elapsed_time = time.time() - start_time
        
    synapse_id = Obj['data_synapses'].iloc[q]['id']
    trimesh.exchange.export.export_mesh(spinemesh, Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q))
        
    return spinemesh   

