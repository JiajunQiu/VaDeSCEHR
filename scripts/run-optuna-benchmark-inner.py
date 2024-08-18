import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


import os
import yaml
import random
import numpy as np
import time
import glob
import subprocess
import utils.utils as utils

from utils.utils import cluster_acc

from pathlib import Path

from utils.plotting import (plot_tsne_by_cluster, plot_group_kaplan_meier, plot_bigroup_kaplan_meier, 
                            plot_tsne_by_survival)
from utils.eval_utils import cindex, calibration, accuracy_metric, cindex_metric
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
from tqdm import tqdm
import math
from typing import Dict, List, Optional
from collections import defaultdict
from optuna.trial import TrialState
from optuna.pruners import BasePruner
import optuna
from sklearn import metrics
from optparse import OptionParser

disc = "Running bayesian hyperparameter optimization"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-c", action="store", type="int", dest="num_clusters", help="number of components for GMM",default=4)
parser.add_option("-n", action="store", type="string", dest="exp_name", help="name of the use_case")
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs")
parser.add_option("-t", action="store", type="string", dest="opt_typ", help="chose optimizer RectifiedAdam or classic Adam",default='Adam')
parser.add_option("-f", action="store", type="string", dest="outer_fold", help="outer_fold")
parser.add_option("-w", action="store", type="string", dest="whole_data", help="whether you want to use with whole dataset or one fold in cross validation. True or False",default='False')
parser.add_option("-s", action="store", type="string", dest="save_model", help="whether you want to save model. True or False",default='False')
parser.add_option("-v", action="store", type="string", dest="surv", help="whether you want to use survival analysis. True or False",default='True')


options, args = parser.parse_args()
num_clusters = options.num_clusters
exp_name = options.exp_name
work_path = options.work_path
opt_typ = options.opt_typ
outer_fold = options.outer_fold
whole = options.whole_data
save = options.save_model
surv=options.surv

if not num_clusters or not exp_name or not work_path or not opt_typ  or not outer_fold:
    parser.print_help()
    sys.exit()

if not os.path.exists(work_path):
    # If the directory doesn't exist, create it
    os.makedirs(work_path)


#script_path=os.path.dirname(os.path.realpath(__file__))


# Get the directory containing the script
script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path=os.path.join(script_path,'data_use_cases/'+exp_name)
temp_path=os.path.join(work_path, 'temp')

if not os.path.exists(temp_path):
    # If the directory doesn't exist, create it
    os.makedirs(temp_path)


class RepeatPruner(BasePruner):
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        trials = study.get_trials(deepcopy=False)
        
        numbers=np.array([t.number for t in trials])
        bool_params= np.array([trial.params==t.params for t in trials]).astype(bool)
        #Don´t evaluate function if another with same params has been/is being evaluated before this one
        if np.sum(bool_params)>1:
            if trial.number>np.min(numbers[bool_params]):
                return True

        return False


def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

    
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

def train(cluster_nums,trial,fold,exp_name,work_path,opt_typ,whole,save):

    tmp_path=os.path.join(temp_path, 'tr_script_'+str(num_clusters)+'_'+str(trial.number)+'@'+str(fold)+'.sh')
#    print(fold,tmp_path)
    out=open(tmp_path,'w')
    print('#!/bin/bash',file=out)
    print('#SBATCH -J optuna',file=out)  
    print('#SBATCH -p gpu',file=out)
    print('#SBATCH -N 1',file=out)             
    print('#SBATCH --gres gpu:1',file=out)
    print('#SBATCH --mem-per-gpu 48G',file=out)
    print('#SBATCH -t 29-12:00',file=out)              
    print('#SBATCH -o '+temp_path+'/job_%A_%a.log',file=out)       
    print('#SBATCH -e '+temp_path+'/job_%A_%a.log',file=out)      
    
    print('python3.9 '+str(os.path.join(script_path,'scripts/run-optuna-fold-benchmark.py'))+' -c '+str(cluster_nums) +' -l '+str(trial.number)+' -f '+str(fold)+' -n '+str(exp_name)+' -p '+str(work_path)+' -t '+str(opt_typ)+' -s '+str(save),file=out)

    out.close()
#    os.system('sbatch '+tmp_path)
    
    output = subprocess.check_output(['sbatch',tmp_path])
    job_id = output.decode().rstrip().split(' ')[-1]

    return job_id

#    print('submited')

def objective_cv(trial):
    setup_seed(seed)
    
    config_path = Path(os.path.join(script_path,'data/pre_trained_model/config_pretraining.yml'))
    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)
    configs['pre_training']['vocab_size1']=len(code1Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size2']=len(code2Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size3']=len(code3Vocab['token2idx'].keys())
    configs['pre_training']['age_vocab_size']=len(ageVocab.keys())

    latent_dim = trial.suggest_categorical("latent_dim", [5,10,15,20])

    weibull_shape = trial.suggest_categorical("weibull_shape", [1,2,3,4,5])
    num_reshape_layers =  trial.suggest_categorical("num_reshape_layers", [1,2,3,4])
    dropout_prob =  trial.suggest_categorical("dropout_prob", [0.1,0.2,0.3,0.4,0.5])

    learning_rate = trial.suggest_categorical("learning_rate", [1e-5,5e-5, 1e-4, 5e-4,1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2,0.1])
    
    if opt_typ == 'RectifiedAdam':
        epoch = trial.suggest_int('epoch',50, 300, step=50)
        warmup_proportion = trial.suggest_categorical("warmup_proportion", [0.01,0.02,0.03])
        
    else:
        epoch = 500
        warmup_proportion = None
        
    if save=='True':
        params={}
        for l in open(os.path.join(work_path,'params'+outer_fold+'.txt')):
            l=l.rstrip()
            t=l.split('\t')
            params[t[0]]=parse_value(t[1])
        epoch = params['epoch']

    configs['training']={}
    configs['training']['latent_dim']=latent_dim
    configs['training']['num_clusters']=num_clusters
    configs['training']['weibull_shape']=weibull_shape
    configs['training']['num_reshape_layers']=num_reshape_layers
    configs['training']['dropout_prob']=dropout_prob
    configs['pre_training']['hidden_dropout_prob']=dropout_prob
    configs['pre_training']['attention_probs_dropout_prob']=dropout_prob

    configs['training']['learning_rate']=learning_rate
    configs['training']['warmup_proportion']=warmup_proportion
    configs['training']['weight_decay']=weight_decay

    configs['training']['epoch']=epoch

    configs['training']['path']=os.path.join(script_path,'data/pre_trained_model')

    configs['training']['opt'] = opt_typ
    configs['training']['learn_prior'] = True
    if surv=='True':
        configs['training']['survival'] = True
        configs['training']['sample_surv'] = True
    elif surv=='False':
        configs['training']['survival'] = False
        configs['training']['sample_surv'] = False
    configs['training']['monte_carlo'] = 1 

    batch_size=100
    job_ids=[]

    if save=='True':
        fold=outer_fold
        steps=math.ceil((len(data_ori[int(fold[0])-1]['code1'])+len(data_ori[int(fold[1])-1]['code1'])+len(data_ori[int(fold[2])-1]['code1'])+len(data_ori[int(fold[3])-1]['code1']))/batch_size)
        configs['training']['t_total'] = steps*epoch
        configs_path = Path(os.path.join(temp_path,'tr_config_'+str(num_clusters)+'_'+str(trial.number)+'@'+str(fold)+'.pkl'))
#        print(fold,configs_path)
        out=open(configs_path,'wb')
        pickle.dump(configs,file=out)
        out.close()
        job_id=train(num_clusters,trial,fold,exp_name,work_path,opt_typ,whole,save)
        job_ids.append(job_id)
        for job_id in tqdm(job_ids):
            flag = True
            while(flag):
                try:
                    output = subprocess.check_output(["squeue","--job",job_id])
                except:
                    flag=False
        return 0

    if trial.should_prune():
        raise optuna.TrialPruned()
    


    idxs='0123'
    for x in range(len(idxs)):
        trn_idxs=idxs[:x]+idxs[x+1:]
        val_idxs=idxs[x]
        fold=outer_fold[int(trn_idxs[0])]+outer_fold[int(trn_idxs[1])]+outer_fold[int(trn_idxs[2])]+'_'+outer_fold[int(val_idxs)]+'_'+outer_fold

        steps=math.ceil((len(data_ori[int(fold[0])-1]['code1'])+len(data_ori[int(fold[1])-1]['code1'])+len(data_ori[int(fold[2])-1]['code1']))/batch_size)
        configs['training']['t_total'] = steps*epoch
        configs_path = Path(os.path.join(temp_path,'tr_config_'+str(num_clusters)+'_'+str(trial.number)+'@'+str(fold)+'.pkl'))
#        print(fold,configs_path)
        out=open(configs_path,'wb')
        pickle.dump(configs,file=out)
        out.close()
        
        job_id=train(num_clusters,trial,fold,exp_name,work_path,opt_typ,whole,save)
        job_ids.append(job_id)
#        print(fold,job_id)
    #check whether the job has started
    for job_id in tqdm(job_ids):
        flag = True
        while(flag):
            try:
                output = subprocess.check_output(["squeue","--job",job_id])
            except:
                flag=False

    nmis=[]
    cis=[]

    for f in glob.glob(os.path.join(temp_path,'tr_loss_'+str(num_clusters)+'_'+str(trial.number)+'@*'+str(outer_fold)+'.txt')):
        tmp_nmis=[]
        tmp_cis=[]
        tmp_epos=[]
        for l in open(f):
            l=l.rstrip()
            epo, nmi, ci=l.split('\t')
            tmp_nmis.append(float(nmi))
            tmp_cis.append(float(ci))
        nmis.append(tmp_nmis)
        cis.append(tmp_cis)
#    print(nmis)

    final_nmi=0
    final_ci=0
    final_epo=0
    for epo in range(len(nmis[0])):
        try:
            tmp1=(nmis[0][epo]+nmis[1][epo]+nmis[2][epo]+nmis[3][epo])/5
            tmp2=(cis[0][epo]+cis[1][epo]+cis[2][epo]+cis[3][epo])/5
            if tmp1>=final_nmi:
                final_nmi=tmp1
                final_ci=tmp2
                final_epo=epo+1
        except:
            break

    trial.set_user_attr('NMI', final_nmi)
    trial.set_user_attr('CI', final_ci)
    trial.set_user_attr('epoch', final_epo)
    os.system('rm -f '+os.path.join(temp_path,'tr_*_'+str(num_clusters)+'_'+str(trial.number)+'@*'+str(outer_fold)+'*'))
    return final_nmi
        


out=open(os.path.join(data_path,'data.pkl'),'rb')
data_ori=pickle.load(out)
out.close()


direction='maximize'



if save=='True':
    params={}
    for l in open(os.path.join(work_path,'params'+outer_fold+'.txt')):
        l=l.rstrip()
        t=l.split('\t')
        params[t[0]]=parse_value(t[1])
    study = optuna.create_study(direction=direction, study_name='train_num_cluster'+str(num_clusters), storage='sqlite:///'+work_path+'/save'+outer_fold+'.db', load_if_exists=True)
    study.enqueue_trial(params)
    study.optimize(objective_cv, n_trials=1)
else:
    study = optuna.create_study(direction=direction, study_name='train_num_cluster'+str(num_clusters), storage='sqlite:///'+work_path+'/train'+outer_fold+'.db', load_if_exists=True,pruner=RepeatPruner())
    study.optimize(objective_cv, n_trials=1)