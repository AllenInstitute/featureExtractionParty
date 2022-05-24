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
#currentdir = "/allen/programs/celltypes/workgroups/em-connectomics/analysis_group/forSharmi/code/featureExtractionParty/external/pointnet_spine_ae"
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
#from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from annotationframeworkclient import infoservice
from itkwidgets import view
#from imageryclient import imagery

from functools import partial
from taskqueue import LocalTaskQueue
from taskqueue import queueable
from annotationframeworkclient import infoservice,FrameworkClient
import tensorflow as tf
from google.cloud import storage
from caveclient import CAVEclient


def savedebugmesh(mesh):
    trimesh.exchange.export.export_mesh(mesh,"test.off")


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
        mydict[u] = submesh
    return mydict
    

def find_closest_component(loc_mesh, x,y,z):

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

def check_if_entry_exists(synid,credentials_path):
    print("checking")
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


def insert_into_PSS_table(synid, pss_vector, cellid, credentials_path):
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

def create_closest_submesh(loc_mesh,seg,x,y,z):
    nfaces = trimesh.proximity.nearby_faces(loc_mesh, [[x,y,z]])
    searchseg = seg[nfaces[0][0]]
    inds = [i for i,val in enumerate(seg) if val==searchseg]
    spine_mesh = loc_mesh.submesh([inds]) 
    return spine_mesh[0]
     
def save_submeshes(submeshes,inds):
    for i,ind in enumerate(inds):
        trimesh.exchange.export.export_mesh(submeshes[ind][0], "debug/pathspine_%d_%d.off"%(ind,i)) 
 
def remove_bad_faces (mesh):
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


def get_mesh(centerpoint, cutout_radius,  segid, bounds, mip, imageclient):
    seg_cutout = imageclient.segmentation_cutout(bounds,mip=mip,root_ids=[segid])
    seg_cutout = np.squeeze(seg_cutout)
    mask = seg_cutout == segid
    mask = np.squeeze(mask)
    mesh = get_trimesh_from_segmask(mask,centerpoint, cutout_radius, mip, imageclient)
    return mesh

def get_trimesh_from_segmask(seg_mask, og_cm, cutout_radius, mip, imageclient):
    
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

def image_data_from_vol_numpy(arr, spacing=[1,1,1], origin=[0,0,0]):
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

def get_local_mesh(Obj,synapse_loc,cellid):
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

def get_local_meshbackup(Obj,synapse_loc):
    pt = np.array(synapse_loc)
    pt[0] = pt[0]/2
    pt[1] = pt[1]/2
    myarray = np.array(Obj['mesh_bounds'])
    
   

    running = True
    while running == True:
        try:
            #print("this is myarray: ",myarray)
            #print(pt)
            #print(Obj['cell_id'])
            mins = pt - myarray
            maxs = pt + myarray
            bbox = cloudvolume.Bbox(mins, maxs)
            #print(mins, maxs, Obj['cell_id'])
            cv = cloudvolume.CloudVolume(Obj['segsource'], use_https=True,secrets=Obj['token'])
            cvmesh=cv.mesh.get(Obj['cell_id'], bounding_box=bbox, use_byte_offsets=True,remove_duplicate_vertices = True)
            
            running = False
        except:
            
            myarray[0] = int(myarray[0]/2)
            myarray[1] = int(myarray[1]/2)
            
        if myarray[0] < 100:
            running = False
        
    
    
    mesh = trimesh_io.Mesh(vertices=cvmesh[Obj['cell_id']].vertices,
                        faces=cvmesh[Obj['cell_id']].faces,
                        process=False)
    
    mesh.faces = remove_bad_faces(mesh)
    is_big = mesh_filters.filter_largest_component(mesh)
    largest_mesh = mesh.apply_mask(is_big)
    ##largest_mesh = mesh
    
    trimesh.smoothing.filter_laplacian(largest_mesh, lamb=0.2, iterations=3, implicit_time_integration=False, volume_constraint=False, laplacian_operator=None)
    #verts = np.array([[0,0.5, 0.7],[0.4,0.5, 0.2],[0.9,0.5, 0.3],[0.8,0.5, 0.5]] )
    #faces = np.array([[1,2,3]])
    #mesh = trimesh_io.Mesh(vertices = verts, faces = faces, process=False)   
    return mesh
    #return 1
    

def get_local_mesh_imagery(Obj,synapse_loc):
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
    
    
def find_path_skel2synapse_cp(loc_mesh,sk,pt):
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

def myprocessingfunctest(Obj,l,q):
    #print ("%d out of %d "%(q,l))
    sc = Obj['synapse_scale'] 
    s = [sc[0]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][0], sc[1]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][1], sc[2]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][2]]
    
    #print("synapse: ", s)
    #loc_mesh = get_segments_for_synapse(Obj,s)
    allmeshes, vertlabels,loc_mesh,pt,sdf,seg,postcellid = get_segments_for_synapse(Obj,s)
    
    #sk = skeletonize.skeletonize_mesh(loc_mesh)
    return loc_mesh


def get_synapse_and_scaled_versions(Obj, q):
    sc = Obj['synapse_scale'] 
    s = [sc[0]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][0], sc[1]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][1], sc[2]*Obj['data_synapses'].iloc[q]['ctr_pt_position'][2]]
    #print(s)
    s_scaled = np.array(s)/[2,2,1]
    s_nm = (s_scaled)*[8,8,40] 
    
    return s, s_nm

def get_synapse_and_scaled_versions_synapseid(Obj, synapse_id):
    
    sc = Obj['synapse_scale']
    client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
    
    data_synapses= client.materialize.query_table('synapses_pni_2',filter_in_dict={'id':['%d'%synapse_id]}, materialization_version = Obj['materialization_version'])
    
    s = [sc[0]*data_synapses.iloc[0]['ctr_pt_position'][0], sc[1]*data_synapses.iloc[0]['ctr_pt_position'][1], sc[2]*data_synapses.iloc[0]['ctr_pt_position'][2]]
    s_scaled = np.array(s)/[2,2,1]
    s_nm = (s_scaled)*[8,8,40] 
    
    return s, s_nm

def get_distance_to_center(Obj,cellid,s_nm):
    #print("this is it")
    #print(Obj['cell_center_of_mass'], )
    #print(cellid)
    if Obj['cell_center_of_mass'] is None:
        
        try:
            client = FrameworkClient(Obj['dataset_name'],auth_token_file=Obj['auth_token_file'])
            #client = Obj['client']
            Obj['cell_center_of_mass'] =client.materialize.query_table('nucleus_detection_v0',filter_in_dict={'pt_root_id':['%d'%cellid]}, materialization_version = Obj['materialization_version'])['pt_position'].values[0]
            
            center = Obj['cell_center_of_mass']*[4,4,40]
            
            #print("These are the points: ", s_nm, center)
            dist_to_center = np.linalg.norm(np.array(s_nm)-center)
            #print("got a center and distance:")
            #print(Obj['cell_center_of_mass'])
            #print(cellid)
            #print(dist_to_center)
            
            if (dist_to_center < 15000): #soma
                Obj["mesh_bounds"] = [200,200,200]
            
        except:
            #This is a floating segment without a nucleus
            dist_to_center = -1000
            Obj["mesh_bounds"] = [200,200,200]
            
    else:
        print(Obj['cell_center_of_mass'])
        
        center = np.array(Obj['cell_center_of_mass'])*np.array([4,4,40])
        #print(center)
        #print(s_nm)
        dist_to_center = np.linalg.norm(np.array(s_nm)-center)
        #print("This is dist to center: ", dist_to_center)
        if (dist_to_center < 15000):
            Obj["mesh_bounds"] = [300,300,300]
    
    return dist_to_center, Obj["mesh_bounds"]

def myprocessingfunc_old(Obj,l,q):
    
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
        allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,postcellid = get_segments_for_synapse(Obj,s,cellid)
        
        #print("Dist to center", dist_to_center)
        
        if dist_to_center < 0: #lone segment 

            spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])
       
        else:
            
            time_start = time.time()
            csg = loc_mesh._create_csgraph()
            ccs = sparse.csgraph.connected_components(csg)
            ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
            large_cc_ids = ccs_u[cc_sizes > 20] # large components in local mesh
            etime = time.time()-time_start

            if (dist_to_center < 15000) & (np.max(cc_sizes) >5000): # soma if its close to the nucleus and component is large

                spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

            else: #dendrite

                sk = skeletonize.skeletonize_mesh(loc_mesh)
                
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
        #trimesh.exchange.export.export_mesh(loc_mesh, Obj['outdir'] + "/locmeshPSS_%d_%d.off"%(synapse_id,q))

    #print("Setting up obj")
    obj = {}
    obj['spinemesh'] = spinemesh
    obj['loc_mesh'] = loc_mesh
    obj['skeleton'] = sk
    obj['pt'] = pt
    print(type(obj))
    return obj

def get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh):
    
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

@queueable
def myprocessingTask(Obj,l,q):
    
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
def myprocessingTask_synapseid(Obj,synapse_id,cellid):
    
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
    print("This is file exists", fname,file_exists, otherexists)
    
    if ((not file_exists) | (Obj['forcerun'] == True)):
    
        s, pt = get_synapse_and_scaled_versions_synapseid(Obj, synapse_id)
        dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
        allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,large_loc_mesh,postcellid = get_segments_for_synapse(Obj,s,cellid)
        print("got all meshes")
        if allmeshes is not None:
            print("Running if")
            if dist_to_center < 0: #lone segment 

                spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

            else:

                spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh)
            print("begin saving to cloud")
            #save to cloud

            spinemesh.vertices = spinemesh.vertices - np.mean(large_loc_mesh.vertices,axis=0)
   
    
            cf = CloudFiles(Obj['cloud_bucket'],secrets =Obj['google_secrets_file'])
            with open(Obj['google_secrets_file'], 'r') as file:
                secretstring = file.read().replace('\n', '')
            print("got secret")
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


    print (synapse_id, "Now starting task" )
    credentials_path = "/usr/local/featureExtractionParty/bigquery_credentials.json"
    flag = check_if_entry_exists(synapse_id,credentials_path)
    
    if not flag:
        tf.reset_default_graph()

        Obj['tensorflow_model'] = importlib.import_module('models.model') # import network module
        
        if not os.path.exists(Obj['pointnet_dump_dir']): 
            os.mkdir(Obj['pointnet_dump_dir'])
        
        Obj['LOG_FOUT'] = open(os.path.join(Obj['pointnet_dump_dir'], 'log_evaluate.txt'), 'w')
        
        outputspinefile = Obj['outdir'] + "/PSS_%d.off"%(synapse_id)
        outputlocmeshfile = Obj['outdir'] + "/locmeshPSS_%d.off"%(synapse_id)
        
        spinemesh = None
        loc_mesh = None
        sk = None
        pt = None
        Obj['synapse_id'] = synapse_id

        
        file_exists = tf.io.gfile.exists(Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id))
        otherexists = os.path.exists(Obj['google_secrets_file'])
        
        fname = Obj['cloud_bucket']+ '%s/PSS_%d.h5'%(Obj['type_of_shape'],synapse_id)
        print("This is file exists", fname,file_exists, otherexists)
        
        if 1==1:
        #if ((not file_exists) | (Obj['forcerun'] == True)):
        
            s, pt = get_synapse_and_scaled_versions_synapseid(Obj, synapse_id)
            dist_to_center,Obj['mesh_bounds'] = get_distance_to_center(Obj,cellid,pt)  
            allmeshes, vertlabels,loc_mesh,other_pt,sdf,seg,large_loc_mesh,postcellid = get_segments_for_synapse(Obj,s,cellid)
            print("got all meshes")
            if allmeshes is not None:
                
                if dist_to_center < 0: #lone segment 

                    spinemesh = create_closest_submesh(loc_mesh,seg,pt[0],pt[1],pt[2])

                else:

                    spinemesh,sk = get_PSS_from_locmesh(pt,s,loc_mesh,dist_to_center,sdf,seg,vertlabels,allmeshes,cellid,Obj,large_loc_mesh)
                print("begin saving to cloud")
                #save to cloud

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
                    

                    print(list(pred_val[0][0]))
                    credentials_path = "/usr/local/featureExtractionParty/bigquery_credentials.json"
                    insert_into_PSS_table(synapse_id, pred_val[0][0].tolist(),postcellid,credentials_path)
                    
                del(spinemesh)
                del(allmeshes)
                del(loc_mesh)
                del(sdf)
                del(seg)
                del(sk)
        del(Obj)
    else:
        print("Record already exists")

            
        
def update_synapses(data_synapses,cell_center_of_mass,threshold):
    dists = np.linalg.norm((np.stack(data_synapses.ctr_pt_position.values) - cell_center_of_mass ) * [4,4,40], axis=1)
    data_synapses['dists'] = dists
    return data_synapses[ data_synapses['dists'] < threshold]




def featureExtractionTask_cell(Obj,cellid,data_synapses):
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
            



    
def myprocessingfunc(Obj,l,q):
    
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

def mySerialProcess(Obj):
 l = len(Obj['rng'])
 for i in Obj['rng']:
        #try:
        if 1==1:
            obj = myprocessingfunc(Obj,l,i)
            #print("this is type of obj: ", type(obj))
        #except: 
            #print("Skipping synapse ", i)
 return obj   
   
def findunprocessed_indices(Obj):
    
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
    
def myParallelTasks(Obj):
    l = len(Obj['rng'])
    
    index = 0
    while index < 3:
        unprocessed_inds = findunprocessed_indices(Obj)
        tq = LocalTaskQueue(parallel=10) # use 5 processes
        #tasks = ( partial(print_task, i) for i in range(2000) ) # NEW SCHOOL
        tasks = (partial(myprocessingTask,Obj,l,q) for q in unprocessed_inds)
        tq.insert_all(tasks) # performs on-line execution (naming is historical)
        index +=1
    
def myParallelProcess_old(Obj):
 l = len(Obj['rng'])
 from multiprocessing import Pool
 from contextlib import closing
 with closing( Pool(45) ) as p:
    partial_process = partial(myprocessingfunc,Obj,l)
    rng = Obj['rng']
    p.map(partial_process,rng)
    
def myParallelProcess(Obj):
     l = len(Obj['rng'])
     from multiprocessing import Process
     procs = []
     for i in Obj['rng']:
        proc = Process(target=myprocessingfunc, args=(Obj,l,i))
        proc.start()
        
     # complete the processes
     for proc in procs:
        proc.join()

def log_string(Obj,out_str):
    Obj['LOG_FOUT'].write(out_str+'\n')
    Obj['LOG_FOUT'].flush()
    #print(out_str)
    

def evaluate(Obj, num_votes=1):
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

def myevalfunc(Obj, ops,is_training, sess,fn):
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
        

    
def eval_one_epoch_cloud_multi(Obj,sess,ops,num_votes=1,topk=1):
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
        
def eval_one_epoch_cloud(Obj, sess, ops, num_votes=1, topk=1):
    
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

def rotate_to_principal_component(vertices):
    pca = decomposition.PCA(n_components=3)
    pca.fit(vertices)
    mainaxis = pca.components_[0]
    yaxis = np.asarray([1,0,0])
    rotation_matrix = rotation_matrix_from_vectors(yaxis,mainaxis)
    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data
              
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_data(Obj,m,num_points):
        #read mesh
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
    


def loadCloudH5File(Obj,filename,num_points):
    
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
def eval_one_epoch(Obj, sess, ops, num_votes=1, topk=1):
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
    #print("Length of features: ")
    #print(features.shape)
    #print(type(features))
    
    
    reducer = umap.UMAP(random_state=20,n_neighbors=20)
    embedding = reducer.fit_transform(features)
    return embedding,reducer
