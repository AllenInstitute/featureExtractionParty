import google.cloud
from google.cloud import bigquery
import tensorflow as tf
import numpy as np
import argparse
import socket
import time
import os
import io
import scipy.misc
import sys
import glob
import umap
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold, preprocessing
from meshparty import trimesh_vtk, mesh_filters
import vtk
import trimesh
from cloudfiles import CloudFiles
import os,sys,inspect
currentdir = "/usr/local/featureExtractionParty/external/pointnet_spine_ae"
sys.path.insert(0,currentdir) 
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
from annotationframeworkclient import infoservice
from itkwidgets import view
from functools import partial
from taskqueue import LocalTaskQueue
from taskqueue import queueable
from annotationframeworkclient import infoservice,FrameworkClient
import tensorflow as tf
from google.cloud import storage
from caveclient import CAVEclient
from featureExtraction.utils import pss_extraction_utils_updated, taskqueue_utils
import psutil
import pandas as pd
import numpy  as np
import concurrent.futures
import datetime



def assign_labels_to_verts(mesh,labels):
    '''
    Given a mesh and a vector of labels corresponding to each of the faces, assign labels to each of the vertices.

    Parameters
    ----------
    mesh: trimesh_io.Mesh
        Input mesh where the number of faces corresponds to the length of labels.
    labels: int list 
        List of the same length as the number of faces of mesh, where each value corresponds to the label of the 
        corresponding face.
    
    Returns
    -------
    vert_labels: int list
        List the length of the number of vertices of the mesh, where each value corresponds to the label 
        of the vertex.
    
    '''
    v = mesh.vertices
    f = mesh.faces
    vert_labels = np.ones(len(v))
    for i,face in enumerate(f):
        vert_labels[face[0]] = labels[i]
        vert_labels[face[1]] = labels[i]
        vert_labels[face[2]] = labels[i]
        
    return vert_labels

def calculate_sdf(mesh_filt):
    '''
    Given a mesh, calculate the sdf (shape distance function) at each vertex and return a vector of these values.

    Parameters
    ----------
    mesh_filt: trimesh_io.Mesh
        Input Mesh

    Returns
    -------
    rs: np.array
        Vector the same length as the number of vertices on the input mesh 
        containing sdf values computed for each vertex.
    '''

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

def check_if_entry_exists(synid,credentials_path):
    '''
    Given a synapse id and the credentials path for big query.

    Parameters
    ----------
    synid: int
        Synapse ID to check entry for in big query table
    credentials_path:
        Path for google credentials
    
    Returns
    -------
    Boolean value True or False if the entry exists in the table.
    
    '''
    sql = """
    select * from  exalted-beanbag-334502.TestDataSet.PSSTable
    where SynapseID = %d
    """%synid
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Construct a BigQuery client object.
    client = bigquery.Client()

    query_job = client.query(sql)  
    numrows = query_job.result().total_rows

    print("This is numrows",numrows)
    if numrows > 0:
        return True
    else:
        return False

def create_closest_submesh(loc_mesh,seg,x,y,z):
    '''
    Given a mesh, a vector containing segmentation information of the mesh and the x,y,z coordinates of any 3D point,
    find the submesh based on the segmentation that is the closest to this point.

    Parameters
    ----------
    loc_mesh: trimesh_io.Mesh
        Input mesh
    seg: int list
        List the length of the number of vertices in loc_mesh, where values correspond to labels of segmentation
        of the mesh into smaller submeshes
    x: int
        3d X coordinate of point to check against
    y: int
        3d Y coordinate of point to check against
    z: int
        3d Z coordinate of point to check against
    Returns
    -------
    spine_mesh: trimesh_io.Mesh
        Submesh (calculated using a precomputed segmentation seg) within loc_mesh that is closest to point [x,y,z]

    '''
    nfaces = trimesh.proximity.nearby_faces(loc_mesh, [[x,y,z]])
    searchseg = seg[nfaces[0][0]]
    inds = [i for i,val in enumerate(seg) if val==searchseg]
    spine_mesh = loc_mesh.submesh([inds]) [0]
    return spine_mesh

def create_data(Obj,m,num_points):
    '''
    Given a processing object, mesh and number of points, sample a fixed number of poitns from the surface mesh 
    and return the point cloud.

    Parameters
    ----------
    Obj: dict
        Processing Object
    m:  trimesh_io.Mesh
        Input Mesh
    num_points: int
        Number of points to sample
    Returns
    -------
    vertices: numpy array
        3D vertices of point cloud sampled on input mesh, with "num_points" number of vertices.
    labels:
        Labels - this is a dummy variable that is needed for PointNet
    
    '''
    try:
        numvertices = len(m.vertices)
        if (numvertices > -100):
            #m = trimesh.Trimesh(vertices,faces)

            cm = np.mean(m.vertices,axis=0)

            m.vertices = m.vertices-cm

            m.vertices = rotate_to_principal_component(m.vertices)

            vertices, fi = trimesh.sample.sample_surface(m,num_points)

            vertices = np.expand_dims(vertices, axis=0)
            labels = np.ndarray([1])
            return (vertices,labels)
        else:
            return None,None
    except:
        return None,None
    
def create_submeshes(mesh,  labels):
    '''
    Given a mesh, and vector containing segmentation labels at the vertices 
    generate all submeshes based on the segmentation labels.

    Parameters
    ----------
    mesh: trimesh_io.Mesh
        Input mesh
    labels: int list
        List the length of the number of vertices in loc_mesh, where values correspond to labels of segmentation
        of the mesh into smaller submeshes
    Returns
    -------
    mydict: dict
        Dict containing all submeshes, where the key for a mesh is the label number in the segmentation label vector

    '''
    allsubmeshes = []
    allkeys = []
    mydict = {}
    unique_labels = list(set(labels))
    for i,u in enumerate(unique_labels):
        inds = [j for j,val in enumerate(labels) if val==u]
        submesh = mesh.submesh([inds])
        mydict[u] = submesh
    return mydict


def eval_one_epoch(Obj, sess, ops, num_votes=1, topk=1):
    
    '''
    Evaluate the processing object, tensorflow session and dict for tensorflow processing,
    read the mesh files, run it through pointnet and save feature in text files on disk.

    Parameters
    ----------
    Obj: dict
        Processing Object
    sess: tf.Session
        Tensorflow session for running pointnet
    ops: dict
        Dict for processing pointnet
    
    '''

    features = []
    filenames = []
    
    is_training = False
    fout = open(os.path.join(Obj['pointnet_dump_dir'], 'pred_label.txt'), 'w')
    for fn in range(len(Obj['pointnet_files'])):
        outfile = Obj['pointnet_files'][fn].replace(".h5","_ae_model_manualV3.txt")
        if (not os.path.exists(outfile)) | (Obj['forcecreatefeatures'] == True):
        #if 1 ==1:
            log_string(Obj,'----'+str(fn)+'----')
            #print(Obj['pointnet_files'][fn])
            current_data, current_label = provider.loadAllOffDataFile(Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
            #current_data,current_label = loadCloudH5File(Obj,Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
            if current_data is not None:
                feed_dict = {ops['pointclouds_pl']: current_data,
                                     ops['labels_pl']: current_label,
                                     ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['embedding']], feed_dict=feed_dict)
                
                np.savetxt(outfile,np.squeeze(pred_val))

def evaluate(Obj, num_votes=1):
    '''
    Given the processing object, extract features and save to disk.

    Parameters
    ----------
    Obj: dict
        Processing Object
       
    '''
    is_training = False
     
    with tf.device('/gpu:'+str(Obj['pointnet_GPU_INDEX'])):
        
        pointclouds_pl, labels_pl = Obj['tensorflow_model'].placeholder_inputs(Obj['pointnet_batch_size'], Obj['pointnet_num_points'])
        #print(pointclouds_pl.shape)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        labels_pl_rep = tf.placeholder(tf.float32,shape=(1))

        # simple model
        pred, end_points = Obj['tensorflow_model'].get_model(pointclouds_pl, is_training_pl)
       #print("Size of embedding")
        #print(pred.shape)
        
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
    
def evaluate_cloud(Obj, num_votes=1):

    '''
    Given the processing object, extract features and save to the cloud.

    Parameters
    ----------
    Obj: dict
        Processing Object
       
    '''
    is_training = False
     
    with tf.device('/gpu:'+str(Obj['pointnet_GPU_INDEX'])):
        
        pointclouds_pl, labels_pl = Obj['tensorflow_model'].placeholder_inputs(Obj['pointnet_batch_size'], Obj['pointnet_num_points'])
        #print(pointclouds_pl.shape)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        labels_pl_rep = tf.placeholder(tf.float32,shape=(1))

        # simple model
        pred, end_points = Obj['tensorflow_model'].get_model(pointclouds_pl, is_training_pl)
       #print("Size of embedding")
        #print(pred.shape)
        
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
    eval_one_epoch_cloud_multi(Obj, sess, ops, num_votes)
    #features,files = eval_one_epoch(FILES, sess, ops, num_votes)
    #features = np.array(features)
    #return features,files

def eval_one_epoch_cloud(Obj, sess, ops, num_votes=1, topk=1):
    '''
    Evaluate the processing object, tensorflow session and dict for tensorflow processing,
    read the mesh files, run it through pointnet and save feature in the cloud.

    Parameters
    ----------
    Obj: dict
        Processing Object
    sess: tf.Session
        Tensorflow session for running pointnet
    ops: dict
        Dict for processing pointnet
    
    '''
    
    is_training = False
    fout = open(os.path.join(Obj['pointnet_dump_dir'], 'pred_label.txt'), 'w')
    for fn in range(len(Obj['pointnet_files'])):
        outfile = Obj['pointnet_files'][fn].replace(".h5","_ae_model_manualV3.json")
        outfile = outfile.replace("PSS", "features/PSS")
        
        #if (not os.path.exists(outfile)) | (Obj['forcecreatefeatures'] == True):
        if 1 ==1:
            log_string(Obj,'----'+str(fn)+'----')
            #print(Obj['pointnet_files'][fn])
            
            current_data,current_label = loadCloudH5File(Obj,Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
            if current_data is not None:
                feed_dict = {ops['pointclouds_pl']: current_data,
                                     ops['labels_pl']: current_label,
                                     ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['embedding']], feed_dict=feed_dict)
                #pred_val = sess.run([ops['pred']], feed_dict=feed_dict)

                
                cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
                with open(Obj['google_secrets_file'], 'r') as file:
                    secretstring = file.read().replace('\n', '')
                #cf = CloudFiles(Obj['cloud_bucket']+ '%s/%d/'%(Obj['type_of_shape'],Obj['nucleus_id']),secrets =secretstring)
                cf = CloudFiles(Obj['cloud_bucket'],secrets =secretstring)
                
                cf.put_json(outfile, content=np.squeeze(pred_val))
                #np.savetxt(outfile,np.squeeze(pred_val))
                
                #print (outfile)
                #features.append(np.squeeze(pred_val))

def eval_one_epoch_cloud_multi(Obj,sess,ops,num_votes=1,topk=1):
    '''
    TEST FUNCTION
    Given the processing object, tensorflow session and dict for tensorflow processing,
    read the mesh files, run it through pointnet and save feature in text files on disk - 
    (Test serial processing and multiprocessing)

    Parameters
    ----------
    Obj: dict
        Processing Object
    sess: tf.Session
        Tensorflow session for running pointnet
    ops: dict
        Dict for processing pointnet
    
    '''

    from multiprocessing import Pool
    from contextlib import closing
    is_training = False
    fout = open(os.path.join(Obj['pointnet_dump_dir'], 'pred_label.txt'), 'w')
    is_training = False
    total = len(Obj['pointnet_files'])
    for r in range(len(Obj['pointnet_files'])):
        print("%d/%d"%(r,total))
        myevalfunc(Obj,ops,is_training,sess,r)
        
    #with closing( Pool(30) ) as p:
    #    partial_process = partial(myevalfunc,Obj,ops,is_training,sess)
    #    rng = range(len(Obj['pointnet_files']))
    #    p.map(partial_process,rng)

def featureExtractionTask_cell(Obj,cellid,data_synapses):
    '''
    Given the processing object object, cell ID and a synapses data frame, calculate the features for the 
    extracted PSS and save on the cloud.

    Parameters
    ----------
    Obj: dict
        Processing object
    cellid: int
        Input Cell id 
    data_synapses: pandas.DataFrame
        Dataframe with synapses
    
    '''

    cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
    with open(Obj['google_secrets_file'], 'r') as file:
        secretstring = file.read().replace('\n', '')
    client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
    #find all files in google bucket
    allfiles = []
    #data_synapses= client.materialize.query_table('synapses_pni_2',filter_in_dict={'post_pt_root_id':['%d'%cellid]}, materialization_version = Obj['materialization_version'])
    
    for index,d in data_synapses.iterrows():
        #print(d.id)
        #cf = CloudFiles(Obj['cloud_bucket']+ '%s/%d/'%(Obj['type_of_shape'],Obj['nucleus_id']),secrets =secretstring)
        #file_exists = cf.exists("PSS_%d.h5"%(d.id))
        #if file_exists:
            #allfiles.append('%s/%d/PSS_%d.h5'%(Obj['type_of_shape'],Obj['nucleus_id'],d.id))
        allfiles.append('%s/PSS_%d.h5'%(Obj['type_of_shape'],d.id))
    print(allfiles)
    Obj['pointnet_files'] = allfiles
    tf.reset_default_graph()
    evaluate_cloud(Obj)
    tf.reset_default_graph()

def find_closest_component(loc_mesh, x,y,z):
    '''
    Given a mesh and the coordinates of a point, find the closest connected component of the mesh to the point. 

    Parameters
    ----------
    loc_mesh: trimesh_io.Mesh
        Input Mesh
    x: int
        X coordinate
    y: int
        Y coordinate
    z: int
        Z coordinate
    
    Returns
    -------
    besta: trimesh_io.Mesh
        Mesh that is the largest connected component 
    
    '''

    print(x,y,z)
    print(np.mean(loc_mesh.vertices,axis=1))    
    allmeshes = trimesh.graph.split(loc_mesh,only_watertight=False)
    dist=None
    besta = None
    for a in allmeshes:
        closest, distance, triangle = trimesh.proximity.closest_point(a, np.array([[x*4,y*4,z*40]])) 
        if (dist is None) :
            besta =a 
            dist = np.abs(distance[0])
        elif  (np.abs(distance[0])< dist) :
            besta =a 
            dist = np.abs(distance[0])
        else:
            t = 10
            #print("Nothing changed")
    return besta

def find_mesh_order(path,vertlabels):
    '''
    Given a path and vertex labels for the whole mesh, find the labels of the meshes that lie on the path.

    Parameters
    ----------
    path: list
        list of indices of vertices that lie on the path of interest
    vertlabels: list
        list of labels of all vertices
    
    Returns
    -------
    pathlabels: list
        list of unique labels of vertices along path (ie: if many points on the path share the same
        label, they are compacted into 1)
    
    '''
    pathlabels = []
    for ip in path:
        pathlabels.append(vertlabels[ip])
    
    unique_path_labels = list(set(pathlabels))
    pathlabels = list(dict.fromkeys(pathlabels)) #get ordered labels
    #save_submeshes(allmeshes,pathlabels)
    return pathlabels

def find_path_skel2synapse_cp(loc_mesh,sk,pt):
    '''
    Given a mesh, the skeleton running on the shaft and a point, find the path starting from the vertex on the mesh 
    that is closest to the point pt to the skeleton sk which runs on the shaft. This function is crucial for the 
    spine extraction.

    Parameters
    ----------
    loc_mesh: trimesh_io.Mesh
        Local Mesh
    sk: skeleton.Skeleton
        Skeleton running on shaft
    pt: numpy array
        coordinatees of point in 3D from which you want to start the path.
    
    Returns
    -------
    path: list of numpy array
        list of points along calculated path from closest vertex on the mesh (to the input point) to the skeleton
    
    '''
    t = get_indices_of_skeleton_verts(loc_mesh, sk)
    sk_inds = [val for i,val in enumerate(t) if not val == -1 ]
    
    if len(sk_inds) < 1:
        return None
    else:
        closest_point, distance, tid = trimesh.proximity.closest_point(loc_mesh,[[pt[0], pt[1], pt[2] ]]) 
        
        pointindex = loc_mesh.faces[tid][0][0]

        dm, preds, sources = sparse.csgraph.dijkstra(
                        loc_mesh.csgraph, False, [pointindex], 
                        min_only=True, return_predecessors=True)
        min_node = np.argmin(dm[sk_inds])

        path = utils.get_path(pointindex, sk_inds[min_node], preds)
        
        
        return path

def findunprocessed_indices(Obj):
    '''
    For local processing, find the synapses whose files on disk that have not been produced / processed yet.

    Parameters
    ----------
    Obj: dict
        processing object
    
    Returns
    -------
    unproc: int list
        indices of synapses in the synapse dataframe that have not been processed yet.
    
    '''
    
    unproc = []
    allindices = Obj['rng']
    
    if not Obj['forcerun'] :
        #print("This is allindices: ", allindices)
        for q in allindices:

            sid = Obj['data_synapses'].iloc[q]['id']

            outfile = Obj['outdir'] + "/PSS_%d_%d.off"%(sid,q)

            if not os.path.exists(outfile):
                unproc.append(q)
    else:
        unproc = allindices
    return unproc

def get_local_mesh(Obj,synapse_loc,cellid):
    '''
    Given the processing object, synapse location and cell id , extract a local mesh around it.

    Parameters
    ----------
    Obj: dict
        Input processing object
    synapse_loc: numpy array
        Synapse coordinates around which local mesh is to be extracted
    cellid: int
        Cell id for which the local mesh is to be extracted
    Returns
    -------
    largest_mesh: Local Mesh around synapse

    '''
    pt = np.array(synapse_loc)
    #print("pt: ", pt)
    pt[0] = pt[0]/2
    pt[1] = pt[1]/2
    
    
    myarray = np.array(Obj['mesh_bounds'])
    
    #print("MY bounds: ", myarray)
    mins = pt - myarray
    maxs = pt + myarray
    
    bbox = cloudvolume.Bbox(mins, maxs)
    #print(bbox)
    #try:
    if 1==1:
        print("Trying to get local mesh")
        print("Segsource; ", Obj['segsource'], Obj['token'])
        cv = cloudvolume.CloudVolume(Obj['segsource'], mip = 2, use_https=True,secrets=Obj['token'])
        print("Yes")
        cvmesh=cv.mesh.get(cellid, bounding_box=bbox, use_byte_offsets=False,remove_duplicate_vertices = True)
        print("got local mesh")
        #print("Got bbox")
        mesh = trimesh_io.Mesh(vertices=cvmesh[cellid].vertices,
                            faces=cvmesh[cellid].faces,
                            process=False)
        #print("removing bad faces")
        mesh.faces = remove_bad_faces(mesh)
        #print("filtering")
        #is_big = mesh_filters.filter_largest_component(mesh)
        #largest_mesh = mesh.apply_mask(is_big)
        #largest_mesh = mesh
        print("going to start closet compoonent")

        
        largest_mesh_trimesh = find_closest_component(mesh, synapse_loc[0], synapse_loc[1], synapse_loc[2])
        largest_mesh = trimesh_io.Mesh(vertices=largest_mesh_trimesh.vertices,
                            faces=largest_mesh_trimesh.faces,
                            process=False)
        #print("smoothing")
        #largest_mesh = mesh
        trimesh.smoothing.filter_laplacian(largest_mesh, lamb=0.2, iterations=4, implicit_time_integration=False, volume_constraint=False, laplacian_operator=None)
        
        #largest_mesh.vertices = largest_mesh.vertices - np.mean(largest_mesh.vertices,axis=0)
        
        #savedebugmesh(mesh)
        #return largest_mesh
        #print("size of mesh: ",largest_mesh.vertices.shape)
    #except:
    else:
        largest_mesh = None
        #print("size of mesh: None")
    
    return largest_mesh
    
def get_local_mesh_imagery(Obj,synapse_loc):
    '''
    Given the processing object and synapse location, get the local mesh by using the segmentation image
    (instead of the precomputed mesh).

    Parameters
    ----------
    Obj: dict
        Processing object
    synapse_loc: numpy array
        3 X 1 array with coordinates of synapse location around which to extract local mesh
    
    Returns
    -------
    M: trimesh_io.Mesh
        Mesh extracted around synapse point

    '''
    imageclient=imagery.ImageryClient(framework_client=Obj['client'])
    pt = np.array(synapse_loc)
    pt[0] = pt[0]/2
    pt[1] = pt[1]/2
    #bounds calculation
    myarray = np.array([1000,1000,500])
    mins = pt - myarray
    maxs = pt + myarray
    bounds = [mins,maxs]
    M = get_mesh(centerpoint=pt, cutout_radius=300,  segid=Obj['cell_id'], bounds=bounds, mip=3, imageclient=imageclient)
    #trimesh.smoothing.filter_humphrey(M, alpha=0.1, beta=0.1, iterations=10, laplacian_operator=None)
    trimesh.smoothing.filter_laplacian(M, lamb=0.9, iterations=3, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)

    return M
   
def get_segments_for_synapse(Obj, synapse_loc,cellid):
    '''
    Given a processing object, synapse location and cellid, find the local mesh, segment it and return a list of
    submeshes.

    Parameters
    ----------
    Obj: dict 
        object containing configuration and input
    synapse_loc: numpy array of size 3
        3d location of synapse
    cellid: int
        cell id to process
    
    Returns
    -------
    allmeshes: list of trimesh_io.mesh
        list of all meshes after segmenting 
    vertlabels: int list
        list of segmentation assignments for vertices of mesh
    loc_mesh: trimesh_io
        local mesh around synapse for segmenting 
    [x,y,z]: int list
        x,y,z coordinates of synapse location
    sdf: float list
        list of sdf values for faces of mesh
    seg: int list
        list of segmentation assignments for faces of mesh
    large_loc_mesh: trimesh_io
        large mesh downloaded around synapse which can be used for skeletonization
    postcellid: int
        postsynaptic cell for which the PSS was extracted.
    
    '''
    x = synapse_loc[0]
    y = synapse_loc[1]
    z = synapse_loc[2]
    
    synapse_id = Obj['synapse_id']
    #Obj['mesh_bounds'] = [500,500,500]
    print("This is synapse loc", synapse_loc)


    ##VERY IMPORTANT - MAKE SURE You ARE EXTRAcTING ON DENDRITE - find post synaptic cellid 
    client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
    table = client.materialize.query_table('synapses_pni_2',filter_in_dict={'id':['%d'%synapse_id]}, materialization_version = Obj['materialization_version'])
    print(type(table))
    print(table.post_pt_root_id.values[0])
    postcellid = table.post_pt_root_id.values[0]
    print("This is postcellid: ", postcellid)
    large_loc_mesh = get_local_mesh(Obj,np.array(synapse_loc),postcellid)

    #print("THIS IS TEH SIZE OF LARGE LOCAL MESH: ", large_loc_mesh.vertices.shape)
    #print("Putting hti sin google")
    #cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
    #with open(Obj['google_secrets_file'], 'r') as file:
    #    secretstring = file.read().replace('\n', '')
    #print("got secret")

    #cf = CloudFiles(Obj['cloud_bucket']+ '%s/'%(Obj['type_of_shape']),secrets =secretstring)
            
     
    #bioloclarge = io.BytesIO()
    #with h5py.File(bioloclarge, "w") as f:
    #    f.create_dataset("vertices", data=large_loc_mesh.vertices, compression="gzip")
    #    f.create_dataset("faces", data=large_loc_mesh.faces, compression="gzip")
    #cf.put("PSSlargelocmesh_%d.h5"%(synapse_id), content=bioloclarge.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
    

    #print("means")
    #print(synapse_loc)
    #print(large_loc_mesh.vertices[0])
    mask = mesh_filters.filter_spatial_distance_from_points(large_loc_mesh, np.array([synapse_loc])*[4,4,40], Obj['local_dist_thresh'])
    loc_mesh = large_loc_mesh.apply_mask(mask)

    #print("THIS IS TEH SIZE OF LOCAL MESH: ", loc_mesh.vertices.shape)
    #bioloc = io.BytesIO()
    #with h5py.File(bioloc, "w") as f:
    #    f.create_dataset("vertices", data=loc_mesh.vertices, compression="gzip")
    #    f.create_dataset("faces", data=loc_mesh.faces, compression="gzip")
    #cf.put("PSSlocmesh_%d.h5"%(synapse_id), content=bioloc.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
    

    #subtracting mean again
    print("This is the mean of large vertices: ", np.mean(large_loc_mesh.vertices,axis=0))
    #loc_mesh.vertices = loc_mesh.vertices - np.mean(large_loc_mesh.vertices,axis=0)
    #large_loc_mesh.vertices = large_loc_mesh.vertices - np.mean(large_loc_mesh.vertices,axis=0)
    
        

    #loc_mesh = get_local_mesh(Obj,np.array(synapse_loc),cellid)
    
    if loc_mesh is None:
        allmeshes=None
        vertlabels = None
        sdf = None
        seg = None
        
    else:
        if (len(loc_mesh.vertices) > 50) & (len(loc_mesh.vertices) < 100000) :
            sdf = cfm.cgal_sdf(loc_mesh.vertices,loc_mesh.faces, Obj['cgal_number_of_rays'],Obj['cgal_cone_angle'])
            seg = cfm.cgal_segmentation(loc_mesh.vertices,loc_mesh.faces, np.asarray(sdf), Obj['cgal_number_of_clusters'], Obj['cgal_smoothing_lambda'])
            vertlabels = assign_labels_to_verts(loc_mesh,seg)    
            allmeshes = create_submeshes(loc_mesh,seg)
        else:
            allmeshes=None
            vertlabels = None
            sdf = None
            seg = None
                    
    return allmeshes,vertlabels,loc_mesh,[x,y,z],sdf,seg, large_loc_mesh,postcellid
    #return loc_mesh

def get_indices_of_skeleton_verts(mesh, sk):
    '''
    Given a mesh and a skeleton, find the indices of the mesh vertices that correspond to skeleton vertices. 
    This was written before the new skeleton updates to meshparty.

    Parameters
    ----------
    mesh: trimesh_io.Mesh
        Input mesh
    sk: skeleton.Skeleton
        Skeleton which lies on this mesh
    
    Returns
    -------
    indices: list
        List of indices of mesh vertices which correspond to skeleton vertices.
    
    '''
    mv = mesh.vertices
    sv = sk.vertices
    mvl = np.ndarray.tolist(mv)
    indices = []
    for s in sv:
        if s in mv:
            indices.append(getind(mvl, s))
    return indices

def getind(vertexlist, point):
    '''
    Given a list of vertices and a single vertex, find its index in the list.

    Parameters
    ----------
    vertexlist: list
        List of vertices
    point: numpy array
        Single input point
    
    Returns
    -------
    i: index of point in the list. Value of -1 if it is not found
    
    '''
    for i in range(0, len(vertexlist)):
        if( (point[0] == vertexlist[i][0]) & (point[1] == vertexlist[i][1]) & (point[2] == vertexlist[i][2])  ):
            return i
    return -1

def get_mesh(centerpoint, cutout_radius,  segid, bounds, mip, imageclient):
    '''
    Get the mesh around a centerpoint using the segmentation image and generating a mesh from that.

    Parameters
    ----------
    centerpoint: numpy array
        Center point around which to generate mesh
    cutout_radius: int
        Radius around cneter point for cutout of mesh
    segid: int
        Segmentation Id of interest
    bounds: numpy array
        2x3 array Bounds for the segmentation image used to generate mask
    mip: int
        Mip level to downsample to
    imageclient: imageclient
        Client to connect to image data
    
    Returns
    -------
    mesh: trimesh_io
        Mesh generated from imagery
    
    '''
    seg_cutout = imageclient.segmentation_cutout(bounds,mip=mip,root_ids=[segid])
    seg_cutout = np.squeeze(seg_cutout)
    mask = seg_cutout == segid
    mask = np.squeeze(mask)
    mesh = get_trimesh_from_segmask(mask,centerpoint, cutout_radius, mip, imageclient)
    return mesh

def get_trimesh_from_segmask(seg_mask, og_cm, cutout_radius, mip, imageclient):
    '''
    Given the Segmentation mask, center point, cutout radius, mip and image client, generate the mesh of the object
    around the center point within the cutout radius.

    Parameters
    ----------
    seg_mask:
        Segmentation mask used to generate mesh
    og_cm:
        Center point around which mesh is captured
    cutout_radius: int
        Radius around center point to extract mesh
    mip: int
        Mip level to use
    imageclient: image client
        Client to connect to image data
    
    Returns
    -------
    new_mesh: trimesh_io
        Mesh generated from imagery
    '''
    
    mip_resolution = imageclient.segmentation_cv.mip_resolution(mip)
    spacing_nm = mip_resolution
    seg_image_data = image_data_from_vol_numpy((255*seg_mask).astype(np.uint8),
                                        spacing = spacing_nm, origin=og_cm-cutout_radius)

    surface =vtk.vtkMarchingCubes()
    surface.SetInputData(seg_image_data)
    surface.ComputeScalarsOff()
    surface.ComputeNormalsOn()
    surface.SetValue(0, 128)
    surface.Update()

    points, tris, edges = trimesh_vtk.poly_to_mesh_components(surface.GetOutput())
    new_mesh = trimesh_io.Mesh(points, tris)
    is_big = mesh_filters.filter_largest_component(new_mesh)
    new_mesh = new_mesh.apply_mask(is_big)

    return new_mesh

def get_indices_of_path(loc_mesh, mesh, point_inds):
    '''
    Given a mesh loc_mesh and a submesh on it mesh, and point_inds which are a subset of points (on a path) 
    in loc_mesh, find the
    Parameters
    ----------
    loc_mesh: trimesh_io.Mesh

    mesh: trimesh_io.Mesh

    point_inds: point indices 
    
    Returns
    -------
    indices: int list
        List of indices of vertex points in loc_mesh that are in mesh.
    
    '''
    mv =mesh.vertices
    mvl = np.ndarray.tolist(mv)
    indices = []
    for p in point_inds:
        point = loc_mesh.vertices[p]
        if point in mv:
            indices.append(p)
    return indices

def get_synapse_and_scaled_versions(Obj, q):
    '''
    Given the processing object and the index of the synapse of interest, return the scaled synapse and synapse in 
    nanometers.

    Parameters
    ----------
    Obj: dict
        Processing object
    q: int
        Index of synapse of interest
    
    Returns
    -------
    s: numpy array
        Scaled synapse
    s_nm: numpy array
        Synapse in nanometers
    
    '''
    sc = Obj['synapse_scale'] 
    s = [sc[0]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][0], sc[1]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][1], sc[2]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][2]]
    #print(s)
    s_scaled = np.array(s)/[2,2,1]
    s_nm = (s_scaled)*[8,8,40] 
    
    return s, s_nm

def get_synapse_and_scaled_versions_synapseid(Obj, synapse_id):
    '''
    Given the processing object and the synapse id, return the scaled synapse and synapse in 
    nanometers.

    Parameters
    ----------
    Obj: dict
        Processing object
    synapse_id: int
        Synapse ID
    
    Returns
    -------
    s: numpy array
        Scaled synapse
    s_nm: numpy array
        Synapse in nanometers
    
    '''
    sc = Obj['synapse_scale']
    client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
    
    data_synapses= client.materialize.query_table('synapses_pni_2',filter_in_dict={'id':['%d'%synapse_id]}, materialization_version = Obj['materialization_version'])
    
    s = [sc[0]*data_synapses.iloc[0]['ctr_pt_position'][0], sc[1]*data_synapses.iloc[0]['ctr_pt_position'][1], sc[2]*data_synapses.iloc[0]['ctr_pt_position'][2]]
    s_scaled = np.array(s)/[2,2,1]
    s_nm = (s_scaled)*[8,8,40] 
    
    return s, s_nm

def get_distance_to_center(Obj,cellid,s_nm):
    '''
    Given the processing object, cell ID and synapse location in nanonmeters, find the distance of the synapse
    to the nucleus center of the cell id.

    Parameters
    ----------
    Obj: dict
        processing object
    cellid: int
        cell id of interest
    s_nm: numpy array
        synapse location in nanometers
    
    Returns
    -------
    dist_to_center: float
        Euclidean distance between synapse and center of cell. If it is a floating segment without a nucleus, 
        this value is returned as -1000
    Obj["mesh_bounds"]: numpy int array 
        Mesh bounds which would change based on the distance to the center (ie: avoid a very 
        large bound range when near the soma)

    
    '''
    #case when you are looking at targets 
    if Obj['cell_center_of_mass'] is None:
        
        try:
            client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
            Obj['cell_center_of_mass'] =client.materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cellid]}, materialization_version = Obj['materialization_version'])['pt_position'].values[0]
            
            center = Obj['cell_center_of_mass']*[4,4,40]
            
            dist_to_center = np.linalg.norm(np.array(s_nm)-center)
            
            if (dist_to_center < 15000): #soma
                Obj["mesh_bounds"] = [200,200,200]
            
        except:
            #This is a floating segment without a nucleus
            dist_to_center = -1000
            Obj["mesh_bounds"] = [200,200,200]

    #case when you are looking at post synaptic shapes on the cell id in question      
    else:
        center = np.array(Obj['cell_center_of_mass'])*np.array([4,4,40])
        dist_to_center = np.linalg.norm(np.array(s_nm)-center)
        if (dist_to_center < 15000):
            Obj["mesh_bounds"] = [300,300,300]
    
    return dist_to_center, Obj["mesh_bounds"]

def get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh):
    '''
    Given all the variables precomputed for PSS extraction, extract it and return it. This function was
    just written to condense code to be more readable in the "myprocessingfunc"s.

    TODO: Clean up some variables

    Parameters
    ----------
    pt: numpy array
        synapse location
    s: numpy array
        scaled synapse point (This one is not really necessary. TODO: remove this variable)
    loc_mesh: trimesh_io.Mesh
        local mesh
    dist_to_center: float
        Distance from synapse to the center of cellid's nucleus
    sdf: list
        List of sdf values for each face of loc_mesh
    seg: list
        List of segmentation labels for each face of loc_mesh
    vertlabels: list
        List of segmentation labels for each vertex of loc_mesh
    allmeshes: trimesh_io.Mesh list 
        List of all submeshes created by the segmentation
    cellid: int
        Cell id for which PSS is being extracted
    Obj: dict
        Processing Object
    large_loc_mesh: trimesh_io.Mesh
        Local mesh for which skeleton is extracted

    Returns
    -------
    spinemesh: trimesh_io.Mesh
        Computed PSS mesh
    sk: skeleton.Skeleton
        Skeleton in local mesh
    '''
    time_start = time.time()
    csg = loc_mesh._create_csgraph()
    ccs = sparse.csgraph.connected_components(csg)
    ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
    large_cc_ids = ccs_u[cc_sizes > 20] # large components in local mesh
    etime = time.time()-time_start

    if (dist_to_center < 15000) & (np.max(cc_sizes) >5000): # soma if its close to the nucleus and component is large
        spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
        sk = None

    else: #dendrite
        #print("dendrite")
        
        #get large loc_mesh
        Obj['mesh_bounds'] = [500,500,500]
        #large_loc_mesh = get_local_mesh(Obj,np.array(s),cellid)
        sk = skeletonize.skeletonize_mesh(large_loc_mesh)

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
            
    return spinemesh,sk

def insert_into_PSS_table(synid, pss_vector, cellid, credentials_path):
    '''
    Given the synapse id, pss 1024 vector, cellid and bigquery credentials path, insert the entry into the 
    bigquery table.

    Parameters
    ----------
    synid:  int
        Synapse ID
    pss_vector: float list
        Vector with PSS feature values
    cellid: int
        Cell id
    credentials_path: string
        Bigquery credentials path
    
    '''
    #credentials_path = '/Users/sharmishtaas/Documents/code/testBigQuery/exalted-beanbag-334502-1a080bb80b37.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Construct a BigQuery client object.
    client = bigquery.Client()


    sql = """
    select * from  exalted-beanbag-334502.TestDataSet.PSSTable
    where SynapseID = %d
    """%synid

    query_job = client.query(sql)  
    numrows = query_job.result().total_rows
    print("This is numrows before insert ",numrows)
    print(type(synid))
    print(type(cellid))
    print(type(pss_vector))
    print(type(pss_vector[0]))
    
    table_id = "exalted-beanbag-334502.TestDataSet.PSSTable"

    rows_to_insert = [
        {u"SynapseID": synid, u"PSSVec": pss_vector,  u"pt_root_id": int(cellid)}
    ]

    #table_id = "exalted-beanbag-334502.TestDataSet.MyNewTable"

    #rows_to_insert = [
    #    {"SynapseID": synid, "pt_root_id": cellid}
    #]

    print("Before inserting")
    errors = client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.

    print("After inserting")
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

def image_data_from_vol_numpy(arr, spacing=[1,1,1], origin=[0,0,0]):
    '''
    Create vtk image data from numpy array

    Parameters
    ----------
    arr: numpy.array
        Input numpy array
    spacing: int list
        Vector describing image spacing in x, y and z. 
    origin: int list
        Coordinate of origin
    
    Returns
    -------
    image_data: vtk.vtkImageData
        vtkImage data containing data from array, and spacing and origin also input.
    
    '''

    #da=trimesh_vtk.numpy_to_vtk(arr.ravel()) #.ravel returns a contiguous flattened array
    
    da = trimesh_vtk.numpy_to_vtk(np.ravel(arr,order='F'))
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(arr.shape)
    image_data.SetExtent(0, arr.shape[0]-1, 0,
                           arr.shape[1]-1, 0, arr.shape[2]-1)
    #assumes 1,1 spacing.
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(origin)
    image_data.GetPointData().SetScalars(da)
    
    return image_data

def log_string(Obj,out_str):
    '''
    Print out logs

    Parameters
    ----------
    Obj: dict
        Dict object containing log information
    out_str: string
        String to be printed
    
    '''
    #Obj['LOG_FOUT'].write(out_str+'\n')
    #Obj['LOG_FOUT'].flush()
    print(out_str)

def loadCloudH5File(Obj,filename,num_points):
    '''
    Load a  mesh file from the cloud, sample the number of points and return a point cloud.

    Parameters
    ----------
    Obj: dict
        Processing object

    filename: string
        H5 file location on the cloud

    num_points: int
        Numbr of vertices to sample
    
    Returns
    -------
    vertices: numpy.array
        3d vertex point cloud for the input shape containing num_points (number of vertices)

    labels: int list
        labels of vertices
    
    '''
    
    #read mesh
    try:
        cloudbucket = Obj['cloud_bucket']
        with open(Obj['google_secrets_file'], 'r') as file:
            secretstring = file.read().replace('\n', '')
        cf = CloudFiles(cloudbucket,secrets =secretstring)
        f = io.BytesIO(cf.get(filename))
        hf = h5py.File(f, 'r')
        vertices= np.array(hf.get('vertices'))
        faces = np.array(hf.get('faces'))
        numvertices = len(vertices)
        if (numvertices > -100):
            m = trimesh.Trimesh(vertices,faces)

            cm = np.mean(m.vertices,axis=0)

            m.vertices = m.vertices-cm

            m.vertices = rotate_to_principal_component(m.vertices)

            vertices, fi = trimesh.sample.sample_surface(m,num_points)

            vertices = np.expand_dims(vertices, axis=0)
            labels = np.ndarray([1])
            return (vertices,labels)
        else:
            return None,None
    except:
        return None,None
   
def loadfeatures(feat_files):

    '''
    Given a list of file names containing features for PSS, load all and return a list of features.

    Parameters
    ----------
    feat_files: string list
        List of feature files (txt format), each containing the feature vector corresponding to one PSS
    
    Returns
    -------
    features: list
        List of features read from feature files
    '''

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

def myParallelTasks(Obj):
    '''
    Given the processing Object, perform PSS extraction locally using taskqueue.
    Parameters
    ----------
    Obj: dict
        processing object containing configuration information
    
    '''
    l = len(Obj['rng'])
    
    index = 0
    while index < 3:
        unprocessed_inds = findunprocessed_indices(Obj)
        tq = LocalTaskQueue(parallel=10) # use 5 processes
        #tasks = ( partial(print_task, i) for i in range(2000) ) # NEW SCHOOL
        tasks = (partial(myprocessingTask,Obj,l,q) for q in unprocessed_inds)
        tq.insert_all(tasks) # performs on-line execution (naming is historical)
        index +=1
    
def myParallelProcess(Obj):
    '''
    Given the processing Object, perform PSS extraction locally using multiprocessing.

    Parameters
    ----------
    Obj: dict
        processing object containing configuration information
    
    '''
    l = len(Obj['rng'])
    from multiprocessing import Process
    procs = []
    for i in Obj['rng']:
        proc = Process(target=myprocessingfunc, args=(Obj,l,i))
        proc.start()
        
    # complete the processes
    for proc in procs:
        proc.join()

def myprocessingfunc(Obj,l,q):
    '''
        Given the processing object, the total number of synapses and the index of the synapse of interest q,
        extract the PSS and save the mesh locally.

        Parameters
        ----------
        Obj: dict
            Input configuration information read into the dict
        l: int
            Total number of synapses
        q: int
            Index of synapse to extract PSS for.
             
    '''
    
    synapse_id = Obj['data_synapses'].iloc[q]['id']
    outputspinefile = Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q)
    outputlocmeshfile = Obj['outdir'] + "/locmeshPSS_%d_%d.off"%(synapse_id,q)
    
    spinemesh = None
    loc_mesh = None
    sk = None
    pt = None
    
    #print("This is forcerun: ", Obj['forcerun'])
    if (not os.path.exists(outputspinefile)) | (Obj['forcerun'] == True):
    
        #print ("%d out of %d "%(q,l))
        #print(Obj['data_synapses'].shape)       
        
        s, pt = get_synapse_and_scaled_versions(Obj, q)
        cellid = Obj['data_synapses'].iloc[q]['post_pt_root_id']   
        dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
        allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg = get_segments_for_synapse(Obj,s,cellid)
        
        if allmeshes is not None:
        
            #print("Dist to center", dist_to_center)

            if dist_to_center < 0: #lone segment 

                spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

            else:
                
                spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj)

            synapse_id = Obj['data_synapses'].iloc[q]['id']
            #print("outputting")
            trimesh.exchange.export.export_mesh(spinemesh, Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q))
            trimesh.exchange.export.export_mesh(loc_mesh, Obj['outdir'] + "/locmeshPSS_%d_%d.off"%(synapse_id,q))
            #cf = CloudFiles(Obj['cloud_bucket'],secrets ='/usr/local/featureExtractionParty/googleservice.json')
            #bio = io.BytesIO()
            #with h5py.File(bio, "w") as f:
            #    f.create_dataset("vertices", data=spinemesh.vertices, compression="gzip")
            #    f.create_dataset("faces", data=spinemesh.faces, compression="gzip")
            
            #cf.put("PSS_%d_%d.h5"%(synapse_id,q), content=bio.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
            
            
        else:
            print("allmeshes in none!!!!!!!!")

    #print("Setting up obj")
    obj = {}
    obj['spinemesh'] = spinemesh
    obj['loc_mesh'] = loc_mesh
    obj['skeleton'] = sk
    obj['pt'] = pt
    
    return obj

@queueable
def myprocessingTask(Obj,l,q):
    '''
        Given the processing object, the total number of synapses and the index of the synapse of interest q,
        extract the PSS and save the mesh in the cloud bucket

        Parameters
        ----------
        Obj: dict
            Input configuration information read into the dict
        l: int
            Total number of synapses
        q: int
            Index of synapse to extract PSS for.
             
    '''
    
    synapse_id = Obj['data_synapses'].iloc[q]['id']
    outputspinefile = Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q)
    outputlocmeshfile = Obj['outdir'] + "/locmeshPSS_%d_%d.off"%(synapse_id,q)
    Obj['synapse_id'] == synapse_id
    spinemesh = None
    loc_mesh = None
    sk = None
    pt = None
    print("Synapse id: ", synapse_id)
    print("This is forcerun: ", Obj['forcerun'])
    if (not os.path.exists(outputspinefile)) | (Obj['forcerun'] == True):
    
        print ("%d out of %d "%(q,l))
        print(Obj['data_synapses'].shape)       
        
        s, pt = get_synapse_and_scaled_versions(Obj, q)
        cellid = Obj['data_synapses'].iloc[q]['post_pt_root_id']   
        dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
        allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,postcellid = get_segments_for_synapse(Obj,s,cellid)
        
        if allmeshes is not None:
        
            #print("Dist to center", dist_to_center)

            if dist_to_center < 0: #lone segment 

                spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

            else:

                spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj)

            synapse_id = Obj['data_synapses'].iloc[q]['id']
            #trimesh.exchange.export.export_mesh(spinemesh, Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q))
            #trimesh.exchange.export.export_mesh(loc_mesh, Obj['outdir'] + "/locmeshPSS_%d_%d.off"%(synapse_id,q))
            
            #print(Obj['outdir'] + "/PSS_%d_%d.off"%(synapse_id,q))
            print("Now saving to cloud")
            print(Obj['cloud_bucket'])
            print("Debug Putting: ", "PSS_%d_%d.h5"%(synapse_id,q))
            cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
            with open(Obj['google_secrets_file'], 'r') as file:
                secretstring = file.read().replace('\n', '')
            cf = CloudFiles(Obj['cloud_bucket'],secrets =secretstring)
            print("Debug Putting: ", "PSS_%d_%d.h5"%(synapse_id,q))
            bio = io.BytesIO()
            with h5py.File(bio, "w") as f:
                f.create_dataset("vertices", data=spinemesh.vertices, compression="gzip")
                f.create_dataset("faces", data=spinemesh.faces, compression="gzip")
            print("Putting: ", "PSS_%d_%d.h5"%(synapse_id,q))
            cf.put("PSS_%d_%d.h5"%(synapse_id,q), content=bio.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)

            
    #print("Setting up obj")
    obj = {}
    obj['spinemesh'] = spinemesh
    obj['loc_mesh'] = loc_mesh
    obj['skeleton'] = sk
    obj['pt'] = pt
    
    return obj

@queueable
def myprocessingTask_cellid_feature(config_file, cellid):
    '''
    Given a config file and cellid, query for all synapses for cellid (threshold of how to filter 
    synapses is in the config file, ie: if we want only the first 60 microns), split the resulting 
    dataframe into 15 splits, parallely process PSS extraction and save result in a pkl file 
    using the cellid.
    TODO: parametrize the split

    Parameters
    ----------
    config_file: string
        Name of config file (JSON)
    cellid: int
        ID of Cell of interest

    '''
    Obj, big_dataframe = taskqueue_utils.create_proc_obj (config_file,cellid)
    fname = '%s/%d.pkl'%(Obj['pss_dataframe_directory'], cellid)    
    
    if not os.path.exists(fname):
        logical    = False
        df_results = []
        
        print("This is the size of big dataframe: ", big_dataframe.shape)
        
        print("This is the size of big dataframe: ", big_dataframe.shape)
        num_procs = Obj['multiproc_n']
        splitted_df = np.array_split(big_dataframe, num_procs)
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = [ executor.submit(process_split_dataframe,df=df, Obj=Obj,cellid = cellid) for df in splitted_df ]
            for result in concurrent.futures.as_completed(results):
                try:
                    df_results.append(result.result())
                except Exception as ex:
                    print(str(ex))
                    pass
        end = time.time()
        print("-------------------------------------------")
        print("PPID %s Completed in %s"%(os.getpid(), round(end-start,2)))
        df_results = pd.concat(df_results)
        print("This is the resulting dataframe shape! ", df_results.shape)
        df_results.to_pickle(fname)
    else:
        print("This file exists!")

@queueable
def myprocessingTask_synapseid(Obj,synapse_id,cellid):
    '''
        Given the processing object, synapse ID and cell id, extract the PSS and output the mesh into the 
        google cloud bucket.

        Parameters
        ----------
        Obj: dict
            Input configuration information read into the dict
        synapse_id: int
            Synapse ID to process
        cellid: int
            Cell ID to extract PSS for
             
    '''

    outputspinefile = Obj['outdir'] + "/PSS_%d.off"%(synapse_id)
    outputlocmeshfile = Obj['outdir'] + "/locmeshPSS_%d.off"%(synapse_id)
    
    spinemesh = None
    loc_mesh = None
    sk = None
    pt = None
    Obj['synapse_id'] = synapse_id
    #file_exists = tf.io.gfile.exists(Obj['cloud_bucket']+ '%s/%d/PSS_%d.h5'%(Obj['type_of_shape'],Obj['nucleus_id'],synapse_id))
    file_exists = tf.io.gfile.exists(Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id))
    otherexists = os.path.exists(Obj['google_secrets_file'])
    
    fname = Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id)
    
    if ((not file_exists) | (Obj['forcerun'] == True)):
    
        s, pt = get_synapse_and_scaled_versions_synapseid(Obj, synapse_id)
        dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
        allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,large_loc_mesh,postcellid = get_segments_for_synapse(Obj,s,cellid)
        if allmeshes is not None:
            if dist_to_center < 0: #lone segment 

                spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

            else:

                spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh)
            #save to cloud

            spinemesh.vertices = spinemesh.vertices - np.mean(large_loc_mesh.vertices,axis=0)
   
    
            cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
            with open(Obj['google_secrets_file'], 'r') as file:
                secretstring = file.read().replace('\n', '')
            #cf = CloudFiles(Obj['cloud_bucket']+ '%s/%d/'%(Obj['type_of_shape'],Obj['nucleus_id']),secrets =secretstring)
            cf = CloudFiles(Obj['cloud_bucket']+ '%s/'%(Obj['type_of_shape']),secrets =secretstring)
            
            bio = io.BytesIO()
            with h5py.File(bio, "w") as f:
                f.create_dataset("vertices", data=spinemesh.vertices, compression="gzip")
                f.create_dataset("faces", data=spinemesh.faces, compression="gzip")
            cf.put("PSS_%d.h5"%(synapse_id), content=bio.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
            
            #bioloc = io.BytesIO()
            #with h5py.File(bioloc, "w") as f:
            #    f.create_dataset("vertices", data=loc_mesh.vertices, compression="gzip")
            #    f.create_dataset("faces", data=loc_mesh.faces, compression="gzip")
            #cf.put("PSSlocmesh_%d.h5"%(synapse_id), content=bioloc.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
            
            #bioloclarge = io.BytesIO()
            #with h5py.File(bioloclarge, "w") as f:
            #    f.create_dataset("vertices", data=large_loc_mesh.vertices, compression="gzip")
            #    f.create_dataset("faces", data=large_loc_mesh.faces, compression="gzip")
            #cf.put("PSSlargelocmesh_%d.h5"%(synapse_id), content=bioloclarge.getvalue(), content_type="application/x-hdf5", compress=None, cache_control=None)
            
            

            del(bio)
            del(cf)
            del(spinemesh)
            del(allmeshes)
            del(loc_mesh)
            del(sdf)
            del(seg)
            del(sk)
    del(Obj)

@queueable
def myprocessingTask_synapseid_feature(Obj,synapse_id,cellid):
    '''
        Given the processing object, synapse ID and cell id, extract the PSS and the 1024 vector and 
        input into the bigquery table. This is the function called for the latest cloud processing.

        Parameters
        ----------
        Obj: dict
            Input configuration information read into the dict
        synapse_id: int
            Synapse ID to process
        cellid: int
            Cell ID to extract PSS for
             
    '''

    print (synapse_id, "Now starting task" )
    #credentials_path = "/usr/local/featureExtractionParty/bigquery_credentials.json"
    #flag = check_if_entry_exists(synapse_id,credentials_path)
    
    if 1==1:
    #if not flag:
        print("Debug 1")
        tf.reset_default_graph()

        Obj['tensorflow_model'] = importlib.import_module('models.model') # import network module
        print("Debug 1.5")
        print(Obj['pointnet_dump_dir'])
        print("Debug 1.7")
        #if not os.path.exists(Obj['pointnet_dump_dir']): 
        #    os.mkdir(Obj['pointnet_dump_dir'])
        print("Debug 2")
        #Obj['LOG_FOUT'] = open(os.path.join(Obj['pointnet_dump_dir'], 'log_evaluate.txt'), 'w')
        
        #outputspinefile = Obj['outdir'] + "/PSS_%d.off"%(synapse_id)
        #outputlocmeshfile = Obj['outdir'] + "/locmeshPSS_%d.off"%(synapse_id)
        print("Debug 3")
        spinemesh = None
        loc_mesh = None
        sk = None
        pt = None
        Obj['synapse_id'] = synapse_id

        print("Debug 4")
        #file_exists = tf.io.gfile.exists(Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id))
        #otherexists = os.path.exists(Obj['google_secrets_file'])
        
        #fname = Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id)
        
        if 1==1:
        #if ((not file_exists) | (Obj['forcerun'] == True)):
            print("Debug 5")
            s, pt = get_synapse_and_scaled_versions_synapseid(Obj, synapse_id)
            print("Debug 6")
            dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
            print("Debug 7")
            allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,large_loc_mesh,postcellid = get_segments_for_synapse(Obj,s,cellid)
            print("Debug 8")
            if allmeshes is not None:
                
                if dist_to_center < 0: #lone segment 

                    spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

                else:

                    spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh)
                

                spinemesh.vertices = spinemesh.vertices - np.mean(large_loc_mesh.vertices,axis=0)
    
                

                with tf.device('/gpu:'+str(Obj['pointnet_GPU_INDEX'])):
            
                    pointclouds_pl, labels_pl = Obj['tensorflow_model'].placeholder_inputs(Obj['pointnet_batch_size'], Obj['pointnet_num_points'])
                    is_training_pl = tf.placeholder(tf.bool, shape=())
                    labels_pl_rep = tf.placeholder(tf.float32,shape=(1))

                    
                    # simple model
                    pred, end_points = Obj['tensorflow_model'].get_model(pointclouds_pl, is_training_pl)
                    
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

                
                current_data, current_label = create_data(Obj,spinemesh,Obj['pointnet_num_points'])
                
                #current_data,current_label = loadCloudH5File(Obj,Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
                if current_data is not None:
                    
                    is_training = False
                    feed_dict = {ops['pointclouds_pl']: current_data,
                                        ops['labels_pl']: current_label,
                                        ops['is_training_pl']: is_training}
                    pred_val = sess.run([ops['embedding']], feed_dict=feed_dict)
                    

                    #credentials_path = "/usr/local/featureExtractionParty/bigquery_credentials.json"
                    #insert_into_PSS_table(synapse_id, pred_val[0][0].tolist(),postcellid,credentials_path)
                    
                del(spinemesh)
                del(allmeshes)
                del(loc_mesh)
                del(sdf)
                del(seg)
                del(sk)
        del(Obj)
    else:
        print("Record already exists")

    return pred_val[0][0].tolist()

def mySerialProcess(Obj):

 '''
    Given the processing object extract the PSS for all synapses in the Object and output the meshes into the 
    google cloud bucket.

    Parameters
    ----------
    Obj: dict
        Input configuration information read into the dict
            
 '''
 l = len(Obj['rng'])
 for i in Obj['rng']:
        #try:
        if 1==1:
            obj = myprocessingfunc(Obj,l,i)
            #print("this is type of obj: ", type(obj))
        #except: 
            #print("Skipping synapse ", i)
 return obj   

def myevalfunc(Obj, ops,is_training, sess,fn):
    '''
    Given the processing object, tensorflow session and dict for tensorflow processing,
    and one mesh file, run it through pointnet and save feature in the cloud.

    Parameters
    ----------
    Obj: dict
        Processing Object
    sess: tf.Session
        Tensorflow session for running pointnet
    ops: dict
        Dict for processing pointnet
    fn: int
        index for which file to process
    is_training: bool
        Flag to determine if the data is for training or not. In the evaluation, it is always false 
        and used only to compute the feature.
    
    '''

    log_string(Obj,'----'+str(fn)+'----')

    
    outfile = Obj['pointnet_files'][fn].replace(".h5","_ae_model_manualV3.json")
    outfile = outfile.replace("PSS", "features/PSS")
        
        
    current_data,current_label = loadCloudH5File(Obj,Obj['pointnet_files'][fn], Obj['pointnet_num_points'])
    if current_data is not None:
        feed_dict = {ops['pointclouds_pl']: current_data,
                             ops['labels_pl']: current_label,
                             ops['is_training_pl']: is_training}
        pred_val = sess.run([ops['embedding']], feed_dict=feed_dict)
        


        cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
        with open(Obj['google_secrets_file'], 'r') as file:
            secretstring = file.read().replace('\n', '')
        cf = CloudFiles(Obj['cloud_bucket'],secrets =secretstring)

        cf.put_json(outfile, content=np.squeeze(pred_val))

def process_split_dataframe(df,Obj,cellid):
    '''
    Given a dataframe, extract the PSS, calculate the 1024 feature and create a dataframe 
    with just the synapse ID and PSS feature

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing synapse information including Synapse ID and others queried from db
    Obj: dict
        Processing Object
    cellid: int
        ID of Cell of interest

    Returns
    -------
    df1: pandas.DataFrame
        Dataframe where each row has a synapse ID and corresponding PSS feature vector
    '''

    df1 = df[['id','post_pt_root_id']]

    pid  = os.getpid()
    ppid = os.getppid()
    start = time.time()
    print("PPID %s->%s Started"%(ppid,pid))

    features = []
    for synapseid in df1['id']:
        try:
            f = myprocessingTask_synapseid_feature(Obj,synapseid,cellid)
        except:
            f = []
        features.append(f)
    df1['PSSfeatures'] = features
    
    stop  = time.time()
    completed_in  = round(stop - start,2)
    return(df1)

def reduce_features_spines(features):
    '''
    Given a list of features, calculate the UMAP fit to 2 dimensions.

    Parameters
    ----------
    features: list
        list of feature vectors (size 1024 for PSS features but would work on any size)

    Returns
    -------
    embedding : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.

    reducer: umap.UMAP
        UMAP model computed from the input dataset
    '''
    
    
    reducer = umap.UMAP(random_state=20,n_neighbors=20)
    embedding = reducer.fit_transform(features)
    return embedding,reducer

def remove_bad_faces (mesh):
    '''
    Remove faces from a mesh with duplicate vertices.

    Parameters
    ----------
    mesh: trimesh_io.mesh
        input mesh object 
    
    Returns
    -------
    allfaces: list
        list of all faces that are valid. (where duplicate vertices have been removed)
    
    '''

    allfaces = []
    for f in  mesh.faces:
        if f[0] == f[1]:
            x=1
        elif (f[1] == f[2]):
            x=1
        elif f[0] == f[2]:
            x=1
        else:
            allfaces.append(list(f))
    return np.array(allfaces)

def rotate_to_principal_component(vertices):
    '''
    Given vertices of a shape, rotate the shape to align with its principal component.

    Parameters
    ----------
    vertices: numpy.array
        Nx3 shaped matrix containing vertices of a 3D shape

    Returns
    -------
    rotated_data: numpy.array
        Nx3 shaped matrix where shape is rotated to the principal component
    
    '''
    pca = decomposition.PCA(n_components=3)
    pca.fit(vertices)
    mainaxis = pca.components_[0]
    yaxis = np.asarray([1,0,0])
    rotation_matrix = rotation_matrix_from_vectors(yaxis,mainaxis)
    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data
              
def rotation_matrix_from_vectors(vec1, vec2):

    '''
    Find the rotation matrix that aligns vec1 (source) to vec2 (destination).

    Parameters
    ----------
    vec1: numpy.array
        A 3d "source" vector
    vec2: numpy.array
        A 3d "destination" vector

    Returns
    -------
    mat: numpy.array
        A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    '''

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def savedebugmesh(mesh):
    '''
    Given a mesh, save it as test.off.

    Parameters
    ----------
    mesh: trimesh_io.Mesh
        input mesh

    '''
    trimesh.exchange.export.export_mesh(mesh,"test.off")

def save_submeshes(submeshes,inds):
    '''
    Given a list of submeshes, save all in the debug directory.

    Parameters
    ----------
    submeshes: list
        list of mesh objects 
    inds:
        indices to be saved
    
    '''

    for i,ind in enumerate(inds):
        trimesh.exchange.export.export_mesh(submeshes[ind][0], "debug/pathspine_%d_%d.off"%(ind,i)) 

def update_synapses(data_synapses,cell_center_of_mass,threshold):
    '''
    Given a dataframe containing synapses, the cell center of mass and a threshold 
    for distance from the center of mass, find all synapses within this radius and
    return them.

    Parameters
    ----------
    data_synapses: pandas.DataFrame
          Dataframe containing synapse ids and all the information for a synapse
    cell_center_of_mass: numpy.array
          3x1 numpy array for the coordinates of the center of mass of the nucleus of the cell
    threshold : int
          Value specifying radius within which to filter synapses
    
    Returns
    -------
    data_synapses: pandas.DataFrame
          Dataframe containing only synapses within the threshold radius
    '''
    dists = np.linalg.norm((np.stack(data_synapses.ctr_pt_position.values) - cell_center_of_mass ) * [4,4,40], axis=1)
    data_synapses['dists'] = dists
    return data_synapses[ data_synapses['dists'] < threshold]
