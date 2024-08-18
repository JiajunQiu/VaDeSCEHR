import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
import numpy as np

import utils.utils as utils

import tensorflow_addons as tfa
# TensorFlow imports
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from models.losses import Losses
from models.model import GMM_Survival
from dataLoader.VAE import VAEData,DataGen
from dataLoader.VAE_benchmark import DataGen as DataGen_benchmark
import pickle
from dataLoader.utils import age_vocab
import tensorflow as tf
import pandas as pd
from optparse import OptionParser
from utils.eval_utils import cindex, calibration, accuracy_metric, cindex_metric,nmi_metric
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


disc = "Running bayesian hyperparameter optimization for each fold"
usage = "usage: %prog [options]"
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-c", action="store", type="string", dest="num_clusters", help="number of components for GMM",default=4)
parser.add_option("-l", action="store", type="int", dest="trial", help="trial number")
parser.add_option("-f", action="store", type="string", dest="fold", help="fold number")
parser.add_option("-n", action="store", type="string", dest="exp_name", help="name of the use_case")
parser.add_option("-p", action="store", type="string", dest="work_path", help="the path to save the outputs")
parser.add_option("-t", action="store", type="string", dest="opt_typ", help="chose optimizer RectifiedAdam or classic Adam")
parser.add_option("-s", action="store", type="string", dest="save_model", help="whether you want to save model")

options, args = parser.parse_args()
num_clusters = options.num_clusters
trial = options.trial
fold = options.fold
exp_name = options.exp_name
work_path = options.work_path
opt_typ = options.opt_typ
save_model = options.save_model


def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

#script_path=os.path.dirname(os.path.realpath(__file__))


# Get the directory containing the script
script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
temp_path=os.path.join(work_path, 'temp')   
data_path=os.path.join(script_path,'data_use_cases/'+exp_name)
config_path = os.path.join(temp_path,'tr_config_'+str(num_clusters)+'_'+str(trial)+'@'+str(fold)+'.pkl')
configs=pickle.load(open(config_path,'rb'))


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

survt=[]
for fd in range(5):
    k ='eventtime'
    fil=os.path.join(data_path,'UKB_EHR_diagnosis_'+k+'_'+str(fd)+'.tsv')
    for l in open(fil):
        l=l.rstrip().split('\t')
        survt.append(int(l[1]))

def define_model_and_opt(configs):
    num_training_steps = configs['training']['t_total']    
    learning_rate = configs['training']['learning_rate']
    warmup_proportion = configs['training']['warmup_proportion']
    weight_decay = configs['training']['weight_decay']
#    setup_seed(seed)

    # Construct the model & optimizer
    model = GMM_Survival(configs)
    if configs['training']['opt']=='RectifiedAdam':
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate,total_steps=num_training_steps,warmup_proportion=warmup_proportion,min_lr=learning_rate/10,weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=weight_decay)
        
    # Use survival times during training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    return model,optimizer

def train(configs,trn_loader,val_loader,save=None):
    losses = Losses(configs)
    rec_loss = losses.loss_sparse_categorical_crossentropy
    model,opt= define_model_and_opt(configs)

    out=open(os.path.join(temp_path,'tr_loss_'+str(num_clusters)+'_'+str(trial)+'@'+str(fold)+'.txt'),'w')
    
        
    model.compile(opt, loss={"output_1": rec_loss}, metrics={"output_4":nmi_metric,"output_5": cindex_metric})

    history = model.fit(trn_loader, validation_data=val_loader, epochs=configs['training']['epoch'], verbose=2)

    for epo in range(len(history.history['val_output_4_nmi_metric'])):
        print(epo+1,history.history['val_output_4_nmi_metric'][epo],history.history['val_output_5_cindex_metric'][epo],file=out,sep='\t')
    out.close()
        
    if save=='True':
        save_path = os.path.join(work_path, 'trained_model')
        model.save_weights(os.path.join(save_path, 'model_weights_fold'+str(fold)))
setup_seed(seed)

              
out=open(os.path.join(data_path,'data.pkl'),'rb')
data_ori=pickle.load(out)
out.close()

batch_size=100
trn_data=data_concat(data_ori[int(fold[0])-1],data_ori[int(fold[1])-1],data_ori[int(fold[2])-1])
val_data=data_concat(data_ori[int(fold[3])-1])

gen_trn = VAEData(trn_data, code1Vocab['token2idx'], code2Vocab['token2idx'], code3Vocab['token2idx'], ageVocab, code_map,max_len=seq_length)
gen_val = VAEData(val_data, code1Vocab['token2idx'], code2Vocab['token2idx'], code3Vocab['token2idx'], ageVocab, code_map,max_len=seq_length)

gen_trn = DataGen_benchmark(gen_trn, batch_size=batch_size,shuffle=True)
gen_val = DataGen_benchmark(gen_val, batch_size=batch_size,shuffle=True)

train(configs, gen_trn, gen_val,save_model)
