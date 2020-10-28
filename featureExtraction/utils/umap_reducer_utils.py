import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
#sys.path.insert(0,parentdir) 
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import glob
import umap
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold, preprocessing

import glob
from ast import literal_eval
import neuroglancer
import json
import requests
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import trimesh
import h5py
#import pandas
import urllib
import urllib.request
import random
from scipy import stats
import pandas as pd


def read_mesh_off(filename):
    """Reads a mesh's vertices and faces from an off file"""
    assert os.path.isfile(filename)
    with open(filename) as fname:
    	content = fname.readlines()
    nums = content[1].split() # 0 - numvert, 1 - numfaces, 2 - numedges
    v = content[2:2+int(nums[0])]
    f = content[2+int(nums[0]): 2+int(nums[0])+int(nums[1])]

    vertices = [[float(i) for i in x.split()][:3] for x in v]
    faces = [[int(i) for i in x.split()][1:] for x in f]

    vertices = np.array(vertices, dtype=np.float)
    faces = np.array(faces, dtype=np.int) 

    return vertices,faces

def read_mesh_h5(filename):
    """Reads a mesh's vertices, faces and normals from an hdf5 file"""
    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as f:
        vertices = f["vertices"][()]
        faces = f["faces"][()]

        if len(faces.shape) == 1:
            faces = faces.reshape(-1, 3)

        if "normals" in f.keys():
            normals = f["normals"][()]
        else:
            normals = []

        if "link_edges" in f.keys():
            link_edges = f["link_edges"][()]
        else:
            link_edges = None
        
        if "node_mask" in f.keys():
            node_mask = f["node_mask"][()]
        else:
            node_mask = None
    return vertices, faces, normals, link_edges, node_mask

def loadfeatures(feat_files):
    features = []
    #synapse_distances = []
    index = 0
    for f in feat_files:
        print(index)
        index += 1
        v = np.loadtxt(f)
        features.append(v)
        #curdistfilename = f.replace('ae_model_v2.txt','distance.txt')
        #synapse_distances.append(np.loadtxt(curdistfilename))
    return features

def split_inds(inds, ALLFEAT_FILES, jsondata):
    allinds = [[],[],[],[], [], []]
    for i in inds:
        filename = ALLFEAT_FILES[i]
        mykey = filename.split("/")[-1].split("_ae")[0]
        allinds[int(jsondata[mykey])].append(i)
    return allinds


def findcell_inds(ALLFEAT_FILES,cellid):
    inds = []
    for i in range (0,len(ALLFEAT_FILES)):
        #print(ALLFEAT_FILES[i].split('/')[-2])
        if int(cellid) == int(ALLFEAT_FILES[i].split('/')[-2]):
            inds.append(i)
        
    return inds

def reduce_features_spines(features):
    print("Length of features: ")
    print(features.shape)
    print(type(features))
    
    
    reducer = umap.UMAP(random_state=20,n_neighbors=20)
    embedding = reducer.fit_transform(features)
    return reducer,embedding

