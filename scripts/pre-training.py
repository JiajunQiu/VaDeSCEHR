import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from models.model_bert import ModelForMaskedLM,CustomSparseCategoricalAccuracy,masked_sparse_categorical_crossentropy
from dataLoader.utils import age_vocab
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask,seq_padding_multi
from dataLoader.MLM import MLMData,DataGen
import tensorflow_addons as tfa
from sklearn.metrics import precision_score
import logging
import yaml
from pathlib import Path
import sys
from tensorflow.keras.callbacks import Callback
 
np.random.seed(42)
random.seed(42) 
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1' 

data_path = sys.argv[1]
checkpoint_path = sys.argv[2]

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

config_path = Path(os.path.join(script_path,'data/pre_trained_model/config_pretraining.yml'))

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
    return output

def define_model_and_opt(steps_per_epoch,path=None):
    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)
    configs['pre_training']['vocab_size1']=len(code1Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size2']=len(code2Vocab['token2idx'].keys())
    configs['pre_training']['vocab_size3']=len(code3Vocab['token2idx'].keys())
    configs['pre_training']['age_vocab_size']=len(ageVocab.keys())


    learning_rate=configs['pre_training']['learning_rate']
    warmup_proportion=configs['pre_training']['warmup_proportion']
    weight_decay=configs['pre_training']['weight_decay']

    num_epochs=100
 
    model = ModelForMaskedLM(configs['pre_training'])
    num_training_steps = steps_per_epoch * num_epochs

    optimizer = tfa.optimizers.RectifiedAdam(lr=learning_rate,total_steps=num_training_steps,warmup_proportion=warmup_proportion,min_lr=1e-6,weight_decay=weight_decay)

    model.compile(optimizer=optimizer,loss=masked_sparse_categorical_crossentropy,metrics=[CustomSparseCategoricalAccuracy()])
    
    return model,num_epochs

out=open(data_path,'rb')
trn_data=pickle.load(out)
out.close()


K_folds=('34512','45123','51234','12345','23451')

trn_data=data_concat(trn_data[int(K_folds[0][0])-1],trn_data[int(K_folds[0][1])-1],trn_data[int(K_folds[0][2])-1],trn_data[int(K_folds[0][3])-1],trn_data[int(K_folds[0][4])-1])
trn_data = MLMData(trn_data, code1Vocab['token2idx'], code2Vocab['token2idx'], code3Vocab['token2idx'], ageVocab, code_map,max_len=seq_length)


data_gen = DataGen(trn_data, batch_size=100)

model,num_epochs = define_model_and_opt(len(data_gen))


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',
    save_weights_only=False,
    save_best_only=False,
    mode='auto',
    verbose=1,
    period=10)

class EpochNumberStop(Callback):
    def __init__(self, stop_epoch):
        super(Callback, self).__init__()
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch == self.stop_epoch:
            self.model.stop_training = True

# Define epoch number stopping callback
#epoch_number_stop = EpochNumberStop(stop_epoch=5)

# Train the model
history = model.fit(data_gen, epochs=num_epochs,verbose=1,callbacks=[model_checkpoint_callback])

model.save_weights(checkpoint_path)
