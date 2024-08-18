import numpy as np
from dataLoader.utils import seq_padding,position_idx,code_convert,code2index
import tensorflow as tf
import math
import pandas as pd
def seq_padding_multi(tokens, max_len, token2idx=None):
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(i)
    return seq

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, X,batch_size=32,shuffle=True):
        self.batch_size = batch_size
        self.X = X
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            inds = np.arange(len(self.X['position']))
            np.random.shuffle(inds)
            for k in self.X:
                self.X[k]=self.X[k][inds]

    def __getitem__(self, index):
        code = self.X['code'][index * self.batch_size:(index + 1) * self.batch_size]

#        position = self.X['position'][index * self.batch_size:(index + 1) * self.batch_size]
        mask = self.X['mask'][index * self.batch_size:(index + 1) * self.batch_size]

        event = self.X['event'][index * self.batch_size:(index + 1) * self.batch_size]
        time = self.X['time'][index * self.batch_size:(index + 1) * self.batch_size]
        cluster = self.X['cluster'][index * self.batch_size:(index + 1) * self.batch_size]

        y = np.column_stack((time, event, cluster))

        return ((code,mask),y),{"output_1": code, "output_4": y, "output_5": y}
    def __len__(self):
        return math.ceil(len(self.X['time']) / self.batch_size)

def VAEData(dataframe, token2idx, max_len):

    self_vocab = token2idx

    self_max_len = max_len

    self_code = pd.DataFrame(dataframe['input'])
    self_time = pd.DataFrame(dataframe['time'])
    self_event = pd.DataFrame(dataframe['event'])
    self_cluster = pd.DataFrame(dataframe['cluster'])

    Dataset={'code':[],'position':[], 'event':[],'time':[],'cluster':[],'mask':[]}

    for index in range(len(self_code)):
        code = self_code.iloc[index].dropna()

        time = self_time.iloc[index].astype(float)
        event = self_event.iloc[index].astype(int)
        cluster = self_cluster.iloc[index].astype(int)


        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self_max_len)
        mask[code=='PAD']= 0

        # pad age sequence and code sequence
        tokens,code= code2index(code, self_vocab)

#        position = position_idx(tokens)


        Dataset['code'].append(code)
#        Dataset['position'].append(position)  
        Dataset['event'].append(event)  
        Dataset['time'].append(time)  
        Dataset['cluster'].append(cluster) 
        Dataset['mask'].append(mask) 
    
    for k in Dataset:
        Dataset[k]=np.array(Dataset[k])
    
    return Dataset












