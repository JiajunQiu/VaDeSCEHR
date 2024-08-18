import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import yaml
import random
import numpy as np

import utils.utils as utils

from pathlib import Path
from utils.plotting import (plot_tsne_by_cluster, plot_group_kaplan_meier, plot_bigroup_kaplan_meier, plot_umap_by_cluster)
from utils.eval_utils import nmi_metric, accuracy_metric, cindex_metric,cindex, calibration, balanced_cluster_acc
from utils.eval_utils import rae as RAE
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
from models.sim.losses import Losses
from models.sim.model import GMM_Survival
from dataLoader.VAE_sim import VAEData,DataGen
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import pandas as pd
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.keras.backend.set_floatx('float64')
from tensorflow.keras.callbacks import Callback
from typing import Dict, List, Optional
from collections import defaultdict
from optuna.trial import TrialState
from optuna.pruners import BasePruner
import optuna
from optparse import OptionParser
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.config.run_functions_eagerly(True)

import sys

class DualOutput:
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.logfile = open(filename, 'a')

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)

    def flush(self):
        self.stdout.flush()
        self.logfile.flush()

disc = "Running model on simultation data"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs or load trained model")

parser.add_option("-d", action="store", type="string", dest="data_fil", help="the path to the simulation data set")

parser.add_option("-e", action="store", type="string", dest="evaluation", help="whether to evaluate previously trained model")

parser.add_option("-s", action="store", type="string", dest="save_model", help="whether you want to save model")

options, args = parser.parse_args()

work_path = options.work_path
data_fil = options.data_fil
evaluation = options.evaluation
save_model = options.save_model

script_path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(script_path, 'data/simulation/simulation.yml')
with Path(config_path).open(mode='r') as yamlfile:
    configs = yaml.safe_load(yamlfile)
    
if evaluation=='True':
    save_model='False'
    
if not os.path.exists(work_path):
    # If the directory doesn't exist, create it
    os.makedirs(work_path)
    
if save_model=='True':
    save_path = os.path.join(work_path, 'trained_model')

    if not os.path.exists(save_path):
        # If the directory doesn't exist, create it
        os.makedirs(save_path)
        


def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  


# Fix random seed
seed = 42
setup_seed(seed)
max_type_size=3


out=open(data_fil,'rb')
data_ori=pickle.load(out)
out.close()

survt=[]
for fold_idx in range(3):
    k ='time'
    survt.extend(data_ori[fold_idx][k])


corpus=['PAD','UNK']
for l in open(os.path.join(script_path,'data/corpus_code2.txt')):
    l=l.rstrip()
    if l not in corpus:
        corpus.append(l)


idx=list(range(len(corpus)))
code2Vocab={}
code2Vocab['token2idx']=dict(zip(corpus,idx))


def data_concat(*data):
    output={}
    for idx,tmp in enumerate(data):
        if idx==0:
            output=tmp
        else:
            for k in output:
                output[k] = pd.concat([output[k],tmp[k]], axis=0, ignore_index=True)
    output['time']=output['time'].astype(int)/(max(survt)+0.001)
    return output


def define_model_and_opt(configs, t_total=-1):

    # Define reconstruction loss function
    losses = Losses(configs)
    rec_loss = losses.loss_sparse_categorical_crossentropy

    setup_seed(seed)

    # Construct the model & optimizer
    model = GMM_Survival(configs)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=configs['training']['learning_rate'], decay=configs['training']['weight_decay'])
    model.compile(optimizer, loss={"output_1": rec_loss}, metrics={"output_4":[nmi_metric,accuracy_metric],"output_5": cindex_metric})
    # Use survival times during training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    return model

        
def train(configs,data):
    seq_length = configs['pre_training']['seq_length']
    
    configs['pre_training']['vocab_size']=len(code2Vocab['token2idx'].keys())
    
    batch_size=100
    
    data[0]['time']=data[0]['time'].astype(float)/(max(survt)+0.0000000001)

    gen_trn = VAEData(data[0], code2Vocab['token2idx'],seq_length)

    gen_trn = DataGen(gen_trn, batch_size=batch_size,shuffle=False)

    model = define_model_and_opt(configs)
    save_path = os.path.join(work_path, 'trained_model/model_weights')
    
    if evaluation=='True':
      
        data[2]['time']=data[2]['time'].astype(float)/(max(survt)+0.0000000001)

        gen_val = VAEData(data[2], code2Vocab['token2idx'],seq_length)
        gen_val = DataGen(gen_val, batch_size=batch_size,shuffle=False) 
    
        model.load_weights(save_path)
        model.sample_surv = False

        rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas, event_time = model.predict(gen_val)
        risk_scores = np.squeeze(risk_scores)
        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        # Hard cluster assignments
        c_hat = np.argmax(p_c_z, axis=-1)
        ci_v = cindex(t=event_time[:,0], d=event_time[:,1], scores_pred=risk_scores)
        eventtime_fold =np.squeeze(event_time[:,0]).astype(float)
        event_fold =np.squeeze(event_time[:,1]).astype(int)
        cluster_fold =np.squeeze(event_time[:,2]).astype(int)

        rae_nc = RAE(t_pred=t_pred_med[event_fold == 1], t_true=eventtime_fold[event_fold== 1],
                    cens_t=1 - event_fold[event_fold == 1])
        rae_c = RAE(t_pred=t_pred_med[event_fold == 0], t_true=eventtime_fold[event_fold == 0],
                    cens_t=1 - event_fold[event_fold == 0])
#        acc_v = utils.cluster_acc(cluster_fold, c_hat)
        acc_v = balanced_cluster_acc(cluster_fold, c_hat)
        nmi_v = normalized_mutual_info_score(cluster_fold, c_hat)
        ari_v = adjusted_rand_score(cluster_fold, c_hat)
        
        outf2=open(os.path.join(work_path,'performance_on_test_set.txt'),'w')
        print('metrics','mean',file=outf2)
        print('CI',ci_v,file=outf2)
        print('rae_nc',rae_nc,file=outf2)
        print('rae_c',rae_c,file=outf2)
        print('NMI',nmi_v,file=outf2)
        print('ACC',acc_v,file=outf2)
        print('ARI',ari_v,file=outf2)
        outf2.close()

        exp_name='simulation'
        plot_group_kaplan_meier(t=eventtime_fold, d=event_fold, c=c_hat,dir=work_path,experiment_name='pred_'+exp_name)
        plot_umap_by_cluster(X=p_c_z, c=c_hat, font_size=12, seed=42, dir=work_path, postfix='pred_umap_'+exp_name)

    else:
#        fil = open(os.path.join(work_path, 'model_history.txt'),'w')

        sys.stdout = DualOutput(os.path.join(work_path, 'model_history.txt'))
        data[1]['time']=data[1]['time'].astype(float)/(max(survt)+0.0000000001)

        gen_val = VAEData(data[1], code2Vocab['token2idx'],seq_length)
        gen_val = DataGen(gen_val, batch_size=batch_size,shuffle=False) 
        
        history = model.fit(gen_trn, validation_data= gen_val, epochs=configs['training']['max_epoch'], verbose=2)

        if save_model=='True':

            model.save_weights(os.path.join(save_path, 'model_weights'))
        
train(configs,data_ori)


