import numpy as np
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask
import tensorflow as tf
import math
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
        code1 = self.X['code1'][index * self.batch_size:(index + 1) * self.batch_size]
        code2 = self.X['code2'][index * self.batch_size:(index + 1) * self.batch_size]
        code3 = self.X['code3'][index * self.batch_size:(index + 1) * self.batch_size]

        age = self.X['age'][index * self.batch_size:(index + 1) * self.batch_size]
        multi = self.X['multi'][index * self.batch_size:(index + 1) * self.batch_size]
        typ = self.X['typ'][index * self.batch_size:(index + 1) * self.batch_size]

        position = self.X['position'][index * self.batch_size:(index + 1) * self.batch_size]
        mask = self.X['mask'][index * self.batch_size:(index + 1) * self.batch_size]
        label = self.X['label'][index * self.batch_size:(index + 1) * self.batch_size]

        return (code1,code2,code3,age,multi,typ,position,mask),label,label
    def __len__(self):
        return math.ceil(len(self.X['position']) / self.batch_size)

def MLMData(dataframe, token2idx1,  token2idx2, token2idx3, age2idx, code_map, max_len):

    self_vocab1 = token2idx1
    self_vocab2 = token2idx2
    self_vocab3 = token2idx3
    self_max_len = max_len
    self_code1 = dataframe['code1']
    self_code2 = dataframe['code2']
    self_code3 = dataframe['code3']
    self_seg = dataframe['segment']
    self_age = dataframe['age']
    self_typ = dataframe['type']
    self_multi = dataframe['multi']
    self_code_map = code_map
    self_age2idx = age2idx

    Dataset={'code1':[], 'code2':[], 'code3':[], 'age':[], 'multi':[],'typ':[],'position':[], 'mask':[], 'label':[]}

    for index in range(len(self_code1)):
        age = self_age.iloc[index].dropna()[(-max_len+1):]
        code1 = self_code1.iloc[index].dropna()[(-max_len+1):]
        code2 = self_code2.iloc[index].dropna()[(-max_len+1):]
        code3 = self_code3.iloc[index].dropna()[(-max_len+1):]
        seg = self_seg.iloc[index].dropna()[(-max_len+1):]
        multi = self_multi.iloc[index].dropna()[(-max_len+1):]
        typ = self_typ.iloc[index].dropna()[(-max_len+1):]

        age = age.reset_index(drop=True)
        code1 = code1.reset_index(drop=True)
        code2 = code2.reset_index(drop=True)
        code3 = code3.reset_index(drop=True)
        seg = seg.reset_index(drop=True).astype(int)
        seg = seg-seg.iloc[0]
        multi = multi.reset_index(drop=True).astype(int)
        multi = multi-multi.iloc[0]
        typ = typ.reset_index(drop=True).astype(int)

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self_max_len)
        mask[len(code1):] = 0

        # pad age sequence and code sequence
        tokens, code1,code2,code3, label = random_mask(code1,code2,code3, self_vocab1,self_vocab2,self_vocab3,self_code_map)
        age = seq_padding(age, self_max_len, token2idx=self_age2idx)

        seg = seq_padding(seg, self_max_len, symbol=max(seg)+1)
        typ = seq_padding(typ, self_max_len, symbol=2)
        multi = seq_padding_multi(multi, self_max_len)

        # get position code and segment code
        tokens = seq_padding(tokens, self_max_len)
        position = position_idx(tokens)


        # pad code and label
        code1 = seq_padding(code1, self_max_len, symbol=self_vocab1['PAD'])
        code2 = seq_padding(code2, self_max_len, symbol=self_vocab2['PAD'])
        code3 = seq_padding(code3, self_max_len, symbol=self_vocab3['PAD'])
        label = seq_padding(label, self_max_len, symbol=-1)

        Dataset['code1'].append(code1)
        Dataset['code2'].append(code2)
        Dataset['code3'].append(code3)
        Dataset['age'].append(age)
        Dataset['multi'].append(multi)
        Dataset['typ'].append(typ)
        Dataset['position'].append(seg)
        Dataset['mask'].append(mask)
        Dataset['label'].append(label)     
    
    for k in Dataset:
        Dataset[k]=np.array(Dataset[k])
    
    return Dataset


