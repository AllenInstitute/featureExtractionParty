import umap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster, spatial, stats
#from dynamicTreeCut import cutreeHybrid, dynamicTreeCut
import seaborn as sns
import networkx as nx
import phenograph
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_predict
import os
import json
from scipy.stats import wasserstein_distance
from pyemd import emd, emd_with_flow
import os
import dash
import pandas as pd
from dashdataframe import configure_app
from nglui.statebuilder import *
import annotationframeworkclient
from annotationframeworkclient import imagery
import random

#Finished


def distmatfunc(i,j,numshollbins):
    
    DM_E = myobj['Euclidean']
    DM_D = myobj['Depth_Matrix']
    DM_W = myobj['EarthMovers_%dbins'%numshollbins]
    
    i = int(i[0]) 
    j = int(j[0]) 
    return facE*DM_E[i,j] + facD*DM_D[i,j] + facW*DM_W[i,j]


def create_random_colors(numcolors):
    allcolors = []
    for i in range(0, numcolors):
        clr = [ round(random.uniform(0, 1),1),round(random.uniform(0, 1),1),round(random.uniform(0, 1),1)]
        allcolors.append(clr)
        
        
    return allcolors
    
    
