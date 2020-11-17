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
#%matplotlib inline  
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
print("Finished imports")


#parameters

cone_angle = (1/4)*math.pi
number_of_rays = 5
number_of_clusters = 5
smoothing_lambda = 0.3
dist_thresh = 3500

#BASIL DATABASE

dataset_name = 'basil'
data_version = 1
sqlalchemy_database_uri="postgresql://postgres:synapsedb@ibs-forrestc-ux1/postgres"
dl = AnalysisDataLink(dataset_name=dataset_name,
                     sqlalchemy_database_uri=sqlalchemy_database_uri,
                     materialization_version=data_version,
                     verbose=False)


#TENSORFLOW MODEL INIT

BATCH_SIZE = 1
NUM_POINT = 2048
MODEL_PATH = '/usr/local/featureExtractionParty/external/pointnet_spine_ae/log_model_manually_selected_trainingset_40_aligned/best_model_epoch_009.ckpt'
GPU_INDEX = 0
MODEL = importlib.import_module('models.model') # import network module
DUMP_DIR = 'dump'
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
HOSTNAME = socket.gethostname()


#FUNCTIONS

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
   
def get_segments_for_synapse(d_mesh, synapse_loc):
    x = synapse_loc[0]
    y = synapse_loc[1]
    z = synapse_loc[2]
    SynSph = Sphere(center=[x,y,z],radius=100)

    start_time = time.time()
    
    filtpts = mesh_filters.filter_spatial_distance_from_points(d_mesh, [[x,y,z]], dist_thresh)
    #print("inseg 1" , time.time() - start_time)

    
    loc_mesh = d_mesh.apply_mask(filtpts)
    #sdf1 = calculate_sdf(loc_mesh)
    print(loc_mesh.vertices)
    print(loc_mesh.faces)
    
    sdf = cfm.cgal_sdf(loc_mesh.vertices,loc_mesh.faces, number_of_rays,cone_angle)
    seg = cfm.cgal_segmentation(loc_mesh.vertices,loc_mesh.faces, np.asarray(sdf), number_of_clusters, smoothing_lambda)
    
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

def myprocessingfunc(data_synapses,d_mesh,l,q):
    print ("%d out of %d "%(q,l))
    
    #s = [4*data_synapses.iloc[q]['centroid_x'], 4*data_synapses.iloc[q]['centroid_y'], 40*data_synapses.iloc[q]['centroid_z']]
    s = [4*data_synapses.iloc[q]['ctr_pt_position'][0], 4*data_synapses.iloc[q]['ctr_pt_position'][1], 40*data_synapses.iloc[q]['ctr_pt_position'][2]]

    print(s)
    allmeshes, vertlabels,loc_mesh,pt,sdf,seg = get_segments_for_synapse(d_mesh,s)
    save_submeshes(allmeshes,range(0,len(allmeshes)))
    
    
    time_start = time.time()
    csg = loc_mesh._create_csgraph()
    ccs = sparse.csgraph.connected_components(csg)
    ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
    large_cc_ids = ccs_u[cc_sizes > 20]
    etime = time.time()-time_start
    
    #debug
    #trimesh.exchange.export.export_mesh(loc_mesh, "locmesh.off") 
    #trimesh.exchange.export.export_mesh(loc_mesh, outdir + "/locmesh_%d.off"%q) 
    print("These are the conditions: ", len(large_cc_ids), " and ", etime)
    #if (len(large_cc_ids) < 8) & (etime < 0.01):
    print(cc_sizes)
    #if 1==1:
    #if (np.where(cc_sizes>3000)[0].shape[0]<2):
    #if etime < 0.01:
    
    dist_to_center = np.linalg.norm(s-cell_center_of_mass)
    
    print("Distance to center: ", dist_to_center, "cellid = ", cell_id )
    
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
            
            #debug
            #save_submeshes(allmeshes, range(0, len(allmeshes)))
            #save_submeshes(allmeshes,pathlabels)
            
            if len(pathlabels) > 1: #only look at cases where you have more than one segment (those will be either good ones or shafts)
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
            #save_submeshes(spinemeshes,range(0,len(spinemeshes)))
            spinemesh = trimesh.util.concatenate(spinemeshes)
            #debug
            #trimesh.exchange.export.export_mesh(spinemesh, "spinemesh.off") 
            elapsed_time = time.time() - start_time


        #trimesh.exchange.export.export_mesh(spinemesh, outdir + "/spine_%d.off"%q)
        
    
        
    synapse_id = data_synapses.iloc[q]['id']
    trimesh.exchange.export.export_mesh(spinemesh, outdir + "/PSS_%d_%d.off"%(synapse_id,q))
        
    return spinemesh
        
    
def myParallelProcess(ahugearray, data_synapses,d_mesh):
 l = len(ahugearray)
 from multiprocessing import Pool
 from contextlib import closing
 with closing( Pool(20) ) as p:
    partial_process = partial(myprocessingfunc,data_synapses,d_mesh,l)
    p.map(partial_process,rng)
    

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    

def evaluate(FILES, num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        print(pointclouds_pl.shape)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        labels_pl_rep = tf.placeholder(tf.float32,shape=(1))

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
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
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl_rep,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'embedding': end_points['embedding']
           }
    eval_one_epoch(FILES, sess, ops, num_votes)
    #features,files = eval_one_epoch(FILES, sess, ops, num_votes)
    #features = np.array(features)
    #return features,files
    
def eval_one_epoch(FILES, sess, ops, num_votes=1, topk=1):
    features = []
    filenames = []
    
    is_training = False
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(FILES)):
        outfile = FILES[fn].replace(".off","_ae_model_manualV3.txt")
        if not os.path.exists(outfile):
        #if 1 ==1:
            log_string('----'+str(fn)+'----')
            print(FILES[fn])
            current_data, current_label = provider.loadAllOffDataFile(FILES[fn], NUM_POINT)
            
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


#GET ALL BASIL CELL IDs

import pandas as pd
#for all cells
filename = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/basil/analysis_dataframe/neurons_phenograph_cluster.pkl'
neuron_df = pd.read_pickle(filename)
cell_id_list = list(neuron_df['soma_id'])

#for np only
#filename = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/synapse_based_spine_extraction#/cgal_tools/notebooks/ANNOTATIONS/annotations_Forrest1.csv'
#annotations = pd.read_csv(filename) 
#cell_id_list = list(annotations[annotations['Tags']=='excitatory']['Segment IDs'].values)
    

onebyone = False


    

#for cell_id in cell_id_list:
for cell_id in ['925087716873']:   
#for cellindex in range(0,len(cell_id_list)):
    
    
    #print("This is the cell index: %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%cellindex)
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #cell_id = cell_id_list[cellindex]
    
    dirstring = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/%s/PSS*.txt'%cell_id
    myfiles =  glob.glob(dirstring)
    dirstring = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/%s/PSS*.off'%cell_id
    offfiles =  glob.glob(dirstring)
    
    
    if 1==1:
    
        cell_id = int(cell_id)
        print("Cell id : %d"%cell_id)
        outdir = '/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/segmentation/synapse_based/EXPT1/%s'%str(cell_id)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if 1 ==1:

            #Download mesh and skeletonize

            if not os.path.exists(outdir+'/%s_skeleton.h5'%cell_id):
                cv_path='https://storage.googleapis.com/neuroglancer/basil_v0/basil_full/seg-aug'
                service_endpoint = 'http://35.237.202.194/meshing/'
                mm=trimesh_io.MeshMeta(cv_path=cv_path, disk_cache_path=outdir)
                #this is because skeletonization was failing with saved skeletons
                d_mesh = mm.mesh(seg_id=cell_id, merge_large_components=False,remove_duplicate_vertices=True,force_download=False)
                #check mesh size
                meshsize = os.path.getsize('%s/%s.h5'%(outdir,cell_id))
                #if meshsize < 50000000:
                if 1==1:
                    sk = skeletonize.skeletonize_mesh(d_mesh,verbose=False)
                    skeleton_io.write_skeleton_h5(sk, outdir+'/%s_skeleton.h5'%cell_id)

            else:
                cv_path='https://storage.googleapis.com/neuroglancer/basil_v0/basil_full/seg-aug'
                service_endpoint = 'http://35.237.202.194/meshing/'
                mm=trimesh_io.MeshMeta(cv_path=cv_path, disk_cache_path=outdir)
                d_mesh = mm.mesh(seg_id=cell_id, merge_large_components=False)
                sk = skeleton_io.read_skeleton_h5(outdir+'/%s_skeleton.h5'%cell_id)


            meshsize = os.path.getsize('%s/%s.h5'%(outdir,cell_id))
            #if meshsize < 50000000: 


            cell_center_of_mass_list = neuron_df[neuron_df['soma_id']==cell_id]['nucleus_center_mass_nm'].values
            if len(cell_center_of_mass_list) > 0:
                cell_center_of_mass = np.array(cell_center_of_mass_list[0])
                if 1==1:
                    #find all synapses
                    #data_synapses = data.loc[data['postsyn_segid'] == cell_id]
                    data_synapses = dl.query_synapses('pni_synapses_i1', post_ids = [cell_id])
                    print(cell_id)

                    rng = range(0,len(data_synapses))
                    
                    print(rng)
                    if onebyone:
                        print('one by one')
                        #for q in [pair[1]]:
                        for q in rng:
                            print (q)
                            spinemeshnew = myprocessingfunc(data_synapses,d_mesh,1,q)
                    else:
                        #multiprocessing
                        print("This is len: %d, %d"%(len(offfiles), 0.95 *len(data_synapses)))
                        if (len(offfiles) < 1.0*len(data_synapses)):
                            myParallelProcess(rng,data_synapses,d_mesh)

                        #CREATE REPRESENTATION
                        cell_directory=outdir
                        DICT_FILES = glob.glob(cell_directory+'/PSS*.off')
                        tf.reset_default_graph()
                        evaluate(DICT_FILES, num_votes=1)
                        tf.reset_default_graph()