import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import random
import numpy as np
 
import utils.utils as utils

from utils.utils import cluster_acc

from pathlib import Path

from utils.plotting import (plot_tsne_by_cluster, plot_group_kaplan_meier, plot_bigroup_kaplan_meier, plot_umap_by_cluster)
from utils.eval_utils import cindex, calibration, accuracy_metric, cindex_metric, balanced_cluster_acc
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import tensorflow_addons as tfa
# TensorFlow imports
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

# VaDeSC model
from models.losses import Losses
from models.model import GMM_Survival
from dataLoader.VAE import VAEData,DataGen
import matplotlib.pyplot as plt
import pickle
from dataLoader.utils import age_vocab
import tensorflow as tf
import pandas as pd
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from typing import Dict, List, Optional
from collections import defaultdict
from optuna.trial import TrialState
from optuna.pruners import BasePruner
import optuna
from utils.eval_utils import rae as RAE
import warnings
warnings.filterwarnings("ignore")
import logging
from optparse import OptionParser
logging.getLogger("tensorflow").setLevel(logging.ERROR)


disc = "Running evaluation"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-c", action="store", type="int", dest="num_clusters", help="max number of components for GMM to test in bayesian hyperparameter optimization",default=4)
parser.add_option("-n", action="store", type="string", dest="exp_name", help="name of the experiment")
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs")
parser.add_option("-b", action="store", type="string", dest="benchmark", help="whether you run on a benchmark data, aka whether you know the labels. True or False",default='False')
parser.add_option("-w", action="store", type="string", dest="whole_data", help="whether you want to use with whole dataset or one fold in cross validation. True or False",default='False')

options, args = parser.parse_args()
num_clusters = options.num_clusters
exp_name = options.exp_name
work_path = options.work_path
benchmark = options.benchmark
whole= options.whole_data
#script_path=os.path.dirname(os.path.realpath(__file__))


# Get the directory containing the script
script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
temp_path=os.path.join(work_path, 'temp')
data_path=os.path.join(script_path,'data_use_cases/'+exp_name)
save_path = os.path.join(work_path, 'trained_model')

    
if not os.path.exists(temp_path):
    # If the directory doesn't exist, create it
    os.makedirs(temp_path)

def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

def parse_value(value_str):
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    try:
        return [int(x) for x in value_str.strip('[]').split(',')]
    except ValueError:
        pass

    return value_str

# Fix random seed
seed = 42
setup_seed(seed)

seq_length=200
max_type_size=3

corpus1=['MASK','UNK','PAD']
code_map={}
for l in open(os.path.join(script_path,'data/corpus_code1.txt')):
    l=l.rstrip()
    if l not in corpus1:
        corpus1.append(l)
corpus2=['MASK','UNK','PAD']
for l in open(os.path.join(script_path,'data/corpus_code2.txt')):
    l=l.rstrip()
    if l not in corpus2:
        corpus2.append(l)
corpus3=['MASK','UNK','PAD']
for l in open(os.path.join(script_path,'data/corpus_code3.txt')):
    l=l.rstrip()
    t=l.split('\t')
    if t[0] not in corpus3:
        corpus3.append(t[0])
        code_map[t[0]]=(t[1],t[2])



idx=list(range(len(corpus1)))
code1Vocab={}
code1Vocab['token2idx']=dict(zip(corpus1,idx))

idx=list(range(len(corpus2)))
code2Vocab={}
code2Vocab['token2idx']=dict(zip(corpus2,idx))

idx=list(range(len(corpus3)))
code3Vocab={}
code3Vocab['token2idx']=dict(zip(corpus3,idx))

ageVocab, _ = age_vocab(max_age=100, mon=1)


survt=[]
cluster_pids=[]
for fd in range(5):
    tmp=[]
    k ='eventtime'
    fil=os.path.join(data_path,'UKB_EHR_diagnosis_'+k+'_'+str(fd)+'.tsv')
    for l in open(fil):
        l=l.rstrip().split('\t')
        survt.append(int(l[1]))
        tmp.append(l[0])
    cluster_pids.append(tmp)


def data_concat(*data):
    output={}
    for idx,tmp in enumerate(data):
        if idx==0:
            output=tmp
        else:
            for k in output:
                output[k] = pd.concat([output[k],tmp[k]], axis=0, ignore_index=True)
    output['eventtime']=output['eventtime'].astype(int)/(max(survt)+0.001)
    return output



def define_model_and_opt(configs, t_total=-1):

    setup_seed(seed)
    # Construct the model & optimizer
    model = GMM_Survival(configs)

    # Use survival times during training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    return model

        
def eval():
    if whole=='True':
        outf1=open(os.path.join(work_path,'cluster_pids.txt'),'w')
    
    config_path = Path(os.path.join(script_path,'data/pre_trained_model/config_pretraining.yml'))
    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)
    configs['pre_training']['vocab_size1']=len(code1Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size2']=len(code2Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size3']=len(code3Vocab['token2idx'].keys())
    configs['pre_training']['age_vocab_size']=len(ageVocab.keys())

    configs['training']={}
    for l in open(os.path.join(data_path,'params.txt')):
        l=l.rstrip()
        t=l.split('\t')
        configs['training'][t[0]]=parse_value(t[1])

    configs['training']['path']=os.path.join(script_path,'data/pre_trained_model')

    configs['training']['num_clusters']=num_clusters
    configs['training']['learn_prior']= True
    configs['training']['survival']= True
    configs['training']['sample_surv']= True
    configs['training']['monte_carlo']= 1 

    batch_size=100
    K_folds=('34512','45123','51234','12345','23451')

    accs=[]
    nmis=[]
    aris=[]
    
    cis = []
    rae_ncs = []
    rae_cs = []
    
    c_hats=[]
    events=[]
    eventtimes=[]
    z_samples=[]
    p_c_zs = []
    ehr_seqs={}

    selected=[]
    for fold in K_folds:
        fdx=4
        val_data=data_concat(data_ori[int(fold[fdx])-1])

        gen_val = VAEData(val_data, code1Vocab['token2idx'], code2Vocab['token2idx'], code3Vocab['token2idx'], ageVocab, code_map,max_len=seq_length)
        gen_val = DataGen(gen_val, batch_size=batch_size,shuffle=False)
        
        model = define_model_and_opt(configs,t_total=len(gen_val)*configs['training']['epoch'])
        if whole=='True':
            model.load_weights(os.path.join(save_path, 'model_weights_overall'))
        elif whole=='False':
            model.load_weights(os.path.join(save_path, 'model_weights_fold'+str(fold)))
        else:
            print('wrong')
            sys.exit()
            
        model.sample_surv = False

        rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas, event_time = model.predict(gen_val)
        risk_scores = np.squeeze(risk_scores)
        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        # Hard cluster assignments
        c_hat = np.argmax(p_c_z, axis=-1)
        ci = cindex(t=event_time[:,0], d=event_time[:,1], scores_pred=risk_scores)
        eventtime_fold =np.squeeze(event_time[:,0]).astype(float)
        event_fold =np.squeeze(event_time[:,1]).astype(int)
        cluster_fold =np.squeeze(event_time[:,2]).astype(int)
        
        rae_nc = RAE(t_pred=t_pred_med[event_fold == 1], t_true=eventtime_fold[event_fold== 1],
                    cens_t=1 - event_fold[event_fold == 1])
        rae_c = RAE(t_pred=t_pred_med[event_fold == 0], t_true=eventtime_fold[event_fold == 0],
                    cens_t=1 - event_fold[event_fold == 0])
        
        p_c_zs.append(p_c_z)
        cis.append(ci)
        rae_ncs.append(rae_nc)
        rae_cs.append(rae_c)

        c_hats.append(c_hat)

        events.append(event_fold)
        eventtime_fold=eventtime_fold*(max(survt)+0.001)/365
        eventtimes.append(eventtime_fold)
        z_samples.append(z_sample[:, 0])

        if benchmark=='True':
            acc_v = balanced_cluster_acc(cluster_fold, c_hat)
            nmi_v = normalized_mutual_info_score(cluster_fold, c_hat)
            ari_v = adjusted_rand_score(cluster_fold, c_hat)            
            accs.append(acc_v)
            nmis.append(nmi_v)
            aris.append(ari_v)

        if whole=='True':
            for idx,patient in enumerate(cluster_pids[int(fold[fdx])-1]):
#            print(patient,c_hat[idx],event_fold[idx],eventtime_fold[idx],' '.join([str(x) for x in z_samples[0][idx]]),file=outff)
                print(patient,c_hat[idx],event_fold[idx],eventtime_fold[idx],file=outf1)

    if whole=='False':
        outf2=open(os.path.join(work_path,'performance_on_test_set.txt'),'w')
        print('metrics','mean','std',file=outf2)
        print('CI',np.mean(cis),np.std(cis),file=outf2)
        print('rae_nc',np.mean(rae_ncs),np.std(rae_ncs),file=outf2)
        print('rae_c',np.mean(rae_cs),np.std(rae_cs),file=outf2)
        if benchmark=='True':
            print('NMI',np.mean(nmis),np.std(nmis),file=outf2)
            print('ACC',np.mean(accs),np.std(accs),file=outf2)
            print('ARI',np.mean(aris),np.std(aris),file=outf2)
        outf2.close()
    if whole=='True':
        outf1.close()
        c_hats =  np.concatenate(c_hats)
        events = np.concatenate(events)
        eventtimes = np.concatenate(eventtimes)
        z_samples = np.vstack(z_samples)
        p_c_zs = np.concatenate(p_c_zs)
        
        plot_group_kaplan_meier(t=eventtimes, d=events, c=c_hats,dir=work_path,experiment_name='pred_'+exp_name)
        plot_tsne_by_cluster(X=z_samples, c=c_hats, font_size=12, seed=42, dir=work_path, postfix='pred_tsne_'+exp_name)
        plot_umap_by_cluster(X=p_c_zs, c=c_hats, font_size=12, seed=42, dir=work_path, postfix='pred_umap_'+exp_name)
        


out=open(os.path.join(data_path,'data.pkl'),'rb')
data_ori=pickle.load(out)
out.close()

eval()

