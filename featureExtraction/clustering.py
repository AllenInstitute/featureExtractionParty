import umap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib notebook
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
import joblib
from featureExtraction.utils import clustering_utils


def precalculate_distance_matrix(config_file):
    
    with open(config_file) as f:
      globvars = json.load(f)
    
    #create weighting matrix
    cluster_centers = pd.read_pickle(globvars['cluster_center_dictionary'])
    distmat = np.zeros([len(cluster_centers),len(cluster_centers)])
    for i in range(0,len(cluster_centers)):
            for j in range(0,len(cluster_centers)):
                distmat[i,j] = np.linalg.norm(cluster_centers[i]-cluster_centers[j])
    distmat1 = (distmat)/np.max(distmat)
    
    
    
    neuron_df = clustering_utils.get_dataframe(globvars)
    featurenames = clustering_utils.get_feature_names(globvars, neuron_df)
    clustering_utils.calculate_diffs(neuron_df)
    M,emd_vec,M3,maxs = clustering_utils.calculate_Matrix(globvars,featurenames, neuron_df)