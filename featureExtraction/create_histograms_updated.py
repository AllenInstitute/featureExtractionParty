from featureExtraction.utils import create_histograms_utils
import json
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
import pandas as pd
import numpy as np
import joblib
import pickle
from annotationframeworkclient import FrameworkClient
import os

def create_one_histogram(cell_id,cfg,mydict):
    #read and query
    data_synapses = cfg['dl'].query_synapses('pni_synapses_i1', post_ids = [cell_id])
    pss_only = pd.read_pickle(cfg['pss_dataframe_directory'] + 'PSS_UMAP_%d.pkl'%cell_id)
    origpss = pss_only.merge(data_synapses,on='id')
    pss = origpss[origpss['pss_mesh_file'] != '']  
    
    #shape features and bins
    
    labels,score,hist = create_histograms_utils.generate_shape_histograms(pss,cfg['dictionarymodel'])
    
    #sholl
    dists = create_histograms_utils.generate_distance_bins(cfg['neuron_df'], data_synapses,cell_id)
    
    #create histogram
    
    pss_sholl = []
    for lab in range(0,30): #shape bins
        labind = np.where(labels==lab)
        mydist = dists[labind]
        
        sholl = []
        for r in range(len(cfg['sholl_radii_upper'])): #distance bins
            upper = cfg['sholl_radii_upper'][r]
            lower = cfg['sholl_radii_lower'][r]
            sholl.append(len(np.where((mydist<upper) & (mydist>=lower))[0]))
            
        
        pss_sholl.extend(sholl)
    
    #outputfeatures
    mydict['allids'].append(cell_id)
    mydict['allhists'].append(hist)
    mydict['allpreds'].append(labels)
    mydict['allscores'].append(score)
    mydict['allpsssholl'].append(pss_sholl)
    return mydict
    
def create_histograms(config_file, cell_id_list = None):
    with open(config_file) as f:
      cfg = json.load(f)
    
    
    #INITS
    
    cfg['dl'] = AnalysisDataLink(dataset_name=cfg['dataset_name'],
                         sqlalchemy_database_uri=cfg['sqlalchemy_database_uri'],
                         materialization_version=cfg['data_version'],
                         verbose=False)
    cfg['dictionarymodel'] = create_histograms_utils.setup_dictionary(cfg['cluster_center_dictionary'])



    cfg['sholl_radii_lower'],cfg['sholl_radii_upper'] = create_histograms_utils.get_sholl_radii(cfg['sholl_params'])

    cfg['neuron_df'] = pd.read_pickle(cfg['input_cell_db'])
    
    if cell_id_list == None:
        cell_id_list = list(neuron_df['soma_id'])

    mydict = create_histograms_utils.create_empty_dataframe_dict()
    index = 0
    
    #CREATE HISTOGRAMS

    for cell_id in cell_id_list:
        create_one_histogram(cell_id,cfg,mydict)
        index+=1
        
    return mydict


def create_dataframes(config_file,cell_id_list = None):
    with open(config_file) as f:
      cfg = json.load(f)
    
    
    #INITS
    cfg['client'] = FrameworkClient(cfg['dataset_name'],auth_token_file=cfg['auth_token_file'])
    #cfg['dl'] = AnalysisDataLink(dataset_name=cfg['dataset_name'],
    #                     sqlalchemy_database_uri=cfg['sqlalchemy_database_uri'],
    #                     materialization_version=cfg['data_version'],
    #                     verbose=False)
    #cfg['neuron_df'] = pd.read_pickle(cfg['input_cell_db'])
    cfg['neuron_df'] = cfg['client'].materialize.query_table('allen_v1_column_types_v2', filter_in_dict={'pt_root_id':[864691136672628359]})

    
    cfg['reducer'] = joblib.load(open(cfg['pss_2d_umap_reducer_file'], 'rb'))
    
    if cell_id_list == None:
        
        cell_id_list = list(cfg['pt_root_id'])
    
    for cell_id in cell_id_list:
        
        outputdirectory = '%s/%s'%(cfg['pss_dataframe_directory'],cfg['type_of_shape'])
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        outputfile = '%s/PSS_UMAP_%d.pkl'%(outputdirectory,cell_id)
        
        if (not os.path.exists(outputfile)) | (cfg['forcesavedataframe'] == True):

            if cfg['type_of_shape'] == 'postsynaptic':
                data_synapses = cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'post_pt_root_id':['%d'%cell_id]})
            else:
                data_synapses = cfg['client'].materialize.query_table('synapses_pni_2',filter_in_dict={'pre_pt_root_id':['%d'%cell_id]})
            feature_files = create_histograms_utils.read_1024features(str(cell_id),cfg)

            feature_embedding0,feature_embedding1,features,myfiles = create_histograms_utils.classify(feature_files,str(cell_id),data_synapses,cfg)

            print("Saving to ... %s"%outputfile)
            pickle.dump(create_histograms_utils.populate_dataframe(data_synapses, features, feature_embedding0, feature_embedding1, myfiles),open(outputfile, "wb" ))
    
