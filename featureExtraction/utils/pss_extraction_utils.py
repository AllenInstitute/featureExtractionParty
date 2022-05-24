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


#import os,sys,inspect
#currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
#sys.path.insert(0,currentdir) 
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
from itkwidgets import view


def assign_labels_to_verts(mesh,labels):
    v = mesh.vertices
    f = mesh.faces
    vert_labels = np.ones(len(v))
    for i,face in enumerate(f):
        vert_labels[face[0]] = labels[i]
        vert_labels[face[1]] = labels[i]
        vert_labels[face[2]] = labels[i]
        
    return vert_labels

def get_indices_of_skeleton_verts(mesh, sk):
    mv = mesh.vertices
    sv = sk.vertices
    mvl = np.ndarray.tolist(mv)
    indices = []
    for s in sv:
        if s in mv:
            indices.append(getind(mvl, s))
    return indices

def getind(vertexlist, point):
    for i in range(0, len(vertexlist)):
        if( (point[0] == vertexlist[i][0]) & (point[1] == vertexlist[i][1]) & (point[2] == vertexlist[i][2])  ):
            return i
    return -1

def create_submeshes(mesh,  labels):
    allsubmeshes = []
    allkeys = []
    mydict = {}
    unique_labels = list(set(labels))
    for i,u in enumerate(unique_labels):
        inds = [j for j,val in enumerate(labels) if val==u]
        submesh = mesh.submesh([inds])
        #print(u)
        mydict[u] = submesh
    return mydict
    
def create_closest_submesh(loc_mesh,seg,x,y,z):
    nfaces = trimesh.proximity.nearby_faces(loc_mesh, [[x,y,z]])
    searchseg = seg[nfaces[0][0]]
    inds = [i for i,val in enumerate(seg) if val==searchseg]
    spine_mesh = loc_mesh.submesh([inds]) 
    return spine_mesh[0]
     
def save_submeshes(submeshes,inds):
    for i,ind in enumerate(inds):
        trimesh.exchange.export.export_mesh(submeshes[ind][0], "debug/pathspine_%d_%d.off"%(ind,i))    
   
def get_segments_for_synapse(Obj, synapse_loc):
    x = synapse_loc[0]
    y = synapse_loc[1]
    z = synapse_loc[2]
    SynSph = Sphere(center=[x,y,z],radius=100)

    start_time = time.time()
    
    filtpts = mesh_filters.filter_spatial_distance_from_points(Obj['d_mesh'], [[x,y,z]],Obj['local_dist_thresh'])
    #print("inseg 1" , time.time() - start_time)

    
    loc_mesh = Obj['d_mesh'].apply_mask(filtpts)
    #sdf1 = calculate_sdf(loc_mesh)
    print(loc_mesh.vertices)
    print(loc_mesh.faces)
    
    sdf = cfm.cgal_sdf(loc_mesh.vertices,loc_mesh.faces, Obj['cgal_number_of_rays'],Obj['cgal_cone_angle'])
    seg = cfm.cgal_segmentation(loc_mesh.vertices,loc_mesh.faces, np.asarray(sdf), Obj['cgal_number_of_clusters'], Obj['cgal_smoothing_lambda'])
    
    vertlabels = assign_labels_to_verts(loc_mesh,seg)
    
    #create_closest_submesh(loc_mesh,seg)
    allmeshes = create_submeshes(loc_mesh,seg)
    
    return allmeshes,vertlabels,loc_mesh,[x,y,z],sdf,seg

def find_path_skel2synapse_cp(loc_mesh,sk,pt):
    t = get_indices_of_skeleton_verts(loc_mesh, sk)
    sk_inds = [val for i,val in enumerate(t) if not val == -1 ]
    #print(sk_inds)
    if len(sk_inds) < 1:
        return None
    else:
        closest_point, distance, tid = trimesh.proximity.closest_point(loc_mesh,[[pt[0], pt[1], pt[2] ]]) 
        print(pt)
        print(closest_point)
        pointindex = loc_mesh.faces[tid][0][0]

        dm, preds, sources = sparse.csgraph.dijkstra(
                        loc_mesh.csgraph, False, [pointindex], 
                        min_only=True, return_predecessors=True)
        min_node = np.argmin(dm[sk_inds])

        path = utils.get_path(pointindex, sk_inds[min_node], preds)
        print("path")
        for p in path:
            print(loc_mesh.vertices[p])
        return path

def find_mesh_order(path,vertlabels):
    pathlabels = []
    for ip in path:
        pathlabels.append(vertlabels[ip])
    
    unique_path_labels = list(set(pathlabels))
    pathlabels = list(dict.fromkeys(pathlabels)) #get ordered labels
    #save_submeshes(allmeshes,pathlabels)
    return pathlabels

def get_indices_of_path(loc_mesh, mesh, point_inds):
    mv =mesh.vertices
    mvl = np.ndarray.tolist(mv)
    indices = []
    for p in point_inds:
        point = loc_mesh.vertices[p]
        if point in mv:
            indices.append(p)
    return indices

def calculate_sdf(mesh_filt):
    ray_inter = ray_pyembree.RayMeshIntersector(mesh_filt)
    # The first argument below sets the origin of the ray, and I use
    # the vertex normal to move the origin slightly so it doesn't hit at the initial vertex point
    # and I can set multiple hits to False

    # The first argument below sets the origin of the ray, and I use
    # the vertex normal to move the origin slightly so it doesn't hit at the initial vertex point
    # and I can set multiple hits to False

    rs = np.zeros(len(mesh_filt.vertices))
    good_rs = np.full(len(rs), False)

    itr = 0
    while not np.all(good_rs):
        #print(np.sum(~good_rs))
        blank_inds = np.where(~good_rs)[0]
        starts = (mesh_filt.vertices-mesh_filt.vertex_normals)[~good_rs,:]
        vs = (-mesh_filt.vertex_normals+0.001*np.random.rand(*mesh_filt.vertex_normals.shape))[~good_rs,:]

        rtrace = ray_inter.intersects_location(starts, vs, multiple_hits=False)
        # radius values
        if len(rtrace[0])>0:
            rs[blank_inds[rtrace[1]]] = np.linalg.norm(mesh_filt.vertices[rtrace[1]]-rtrace[0], axis=1)
            good_rs[blank_inds[rtrace[1]]]=True
        itr+=1
        if itr>10:
            break
    return rs

def myprocessingfunc(Obj,l,q):
    print ("%d out of %d "%(q,l))
    
    s = [4*Obj['data_synapses'].iloc[q]['ctr_pt_position'][0], 4*Obj['data_synapses'].iloc[q]['ctr_pt_position'][1], 40*Obj['data_synapses'].iloc[q]['ctr_pt_position'][2]]

    print(s)
    allmeshes, vertlabels,loc_mesh,pt,sdf,seg = get_segments_for_synapse(Obj,s)
    
    
    time_start = time.time()
    csg = loc_mesh._create_csgraph()
    ccs = sparse.csgraph.connected_components(csg)
    ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
    large_cc_ids = ccs_u[cc_sizes > 20]
    etime = time.time()-time_start
    
    dist_to_center = np.linalg.norm(s-Obj['cell_center_of_mass'])
    
    print("Distance to center: ", dist_to_center, "cellid = ", Obj['cell_id'] )
    
    if (dist_to_center < 15000) & (np.max(cc_sizes) >5000):
        
        spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
        
    else:
                
        try:
            path = find_path_skel2synapse_cp(loc_mesh,sk,pt)
        except:
            path = None

        if path is None:
            spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
        else:
            start_time = time.time()
            pathlabels = find_mesh_order(path,vertlabels)
            
            if len(pathlabels) > 1: #only cases with more than one segment (those will be either good ones or shafts)
                sdf_verts = assign_labels_to_verts(loc_mesh,sdf)
                sdf_mean = []
                for ind in range(0,len(pathlabels)):
                    lastmesh = allmeshes[pathlabels[ind]][0]
                    t1 = get_indices_of_path(loc_mesh, lastmesh, path)
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
        
    
def myParallelProcess(Obj):
 l = len(Obj['rng'])
 from multiprocessing import Pool
 from contextlib import closing
 with closing( Pool(20) ) as p:
    partial_process = partial(myprocessingfunc,Obj,l)
    rng = Obj['rng']
    p.map(partial_process,rng)
    

def log_string(Obj,out_str):
    Obj['LOG_FOUT'].write(out_str+'\n')
    Obj['LOG_FOUT'].flush()
    print(out_str)
    

def evaluate(Obj, num_votes=1):
    is_training = False
     
    with tf.device('/gpu:'+str(Obj['pointnet_GPU_INDEX'])):
        
        pointclouds_pl, labels_pl = Obj['tensorflow_model'].placeholder_inputs(Obj['pointnet_batch_size'], Obj['pointnet_num_points'])
        print(pointclouds_pl.shape)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        labels_pl_rep = tf.placeholder(tf.float32,shape=(1))

        # simple model
        pred, end_points = Obj['tensorflow_model'].get_model(pointclouds_pl, is_training_pl)
        print("Size of embedding")
        print(pred.shape)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, Obj['pointnet_model_path'])
    log_string(Obj,"Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl_rep,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'embedding': end_points['embedding']
           }
    eval_one_epoch(Obj, sess, ops, num_votes)
    #features,files = eval_one_epoch(FILES, sess, ops, num_votes)
    #features = np.array(features)
    #return features,files
    
def eval_one_epoch(Obj, sess, ops, num_votes=1, topk=1):
    features = []
    filenames = []
    
    is_training = False
    fout = open(os.path.join(Obj['pointnet_dump_dir'], 'pred_label.txt'), 'w')
    for fn in range(len(Obj['pointnet_files'])):
        outfile = Obj['pointnet_files'][fn].replace(".off","_ae_model_manualV3.txt")
        if not os.path.exists(outfile):
        #if 1 ==1:
            log_string(Obj,'----'+str(fn)+'----')
            print(Obj['pointnet_files'][fn])
            current_data, current_label = provider.loadAllOffDataFile(Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
            
            if current_data is not None:
                feed_dict = {ops['pointclouds_pl']: current_data,
                                     ops['labels_pl']: current_label,
                                     ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['embedding']], feed_dict=feed_dict)
                #pred_val = sess.run([ops['pred']], feed_dict=feed_dict)

                np.savetxt(outfile,np.squeeze(pred_val))
                
                print (outfile)
                #features.append(np.squeeze(pred_val))
    
def loadfeatures(feat_files):
    features = []
    #synapse_distances = []
    index = 0
    for f in feat_files:
        index += 1
        v = np.loadtxt(f)
        features.append(v)
        #curdistfilename = f.replace('ae_model_v2.txt','distance.txt')
        #synapse_distances.append(np.loadtxt(curdistfilename))
    return features

def dist2pt (features, pt):
    features = np.array(features)
    pt = np.array(pt)
    dists = np.linalg.norm(features-pt[np.newaxis,:],axis=1) 
    return np.argmin(dists,axis=0)  
 
def reduce_features_spines(features):
    print("Length of features: ")
    print(features.shape)
    print(type(features))
    
    
    reducer = umap.UMAP(random_state=20,n_neighbors=20)
    embedding = reducer.fit_transform(features)
    return embedding,reducer