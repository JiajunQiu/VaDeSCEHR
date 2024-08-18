import numpy as np
from dataLoader.utils import seq_padding,position_idx,code_convert
import tensorflow as tf
import math
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

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
    def __init__(self, X,batch_size=32,shuffle=True,pretrain=False,balance=False):
        self.batch_size=batch_size
        self.balance=balance
        self.X = X
        self.shuffle = shuffle
        self.pretrain=pretrain
        if self.balance:
            unique_clusters, cluster_counts = np.unique(self.X['cluster'], return_counts=True)
            max_count = np.max(cluster_counts)
            self.len_data = math.ceil((max_count*len(unique_clusters)) / self.batch_size)
        else:
            self.len_data = math.ceil(len(self.X['cluster']) / self.batch_size)

        self.kf = StratifiedShuffleSplit(n_splits=self.len_data,test_size=self.batch_size)
        self.first_epoch()

    def first_epoch(self):
        clst=self.X['cluster']
        if self.balance:
            unique_clusters, cluster_counts = np.unique(clst, return_counts=True)
            max_count = np.max(cluster_counts)
        
            self.selected_indices = []
            self.selected_clst = []
            for cluster in unique_clusters:
                cluster_indices = np.where(clst == cluster)[0]
                if len(cluster_indices) == max_count:
                    self.selected_indices.extend(cluster_indices)
                else:
                    self.selected_indices.extend(np.concatenate((cluster_indices, np.random.choice(cluster_indices, size=max_count-len(cluster_indices), replace=True)), axis=None))
                self.selected_clst.extend([cluster]*max_count)
        else:
            self.selected_indices = np.arange(len(self.X['position']))
            self.selected_clst = clst

        self.inds_batch=[]
        for fold, (train_index, test_index) in enumerate(self.kf.split(self.selected_indices,self.selected_clst)):
            ids=[self.selected_indices[i] for i in test_index]
            self.inds_batch.append(ids)

    def on_epoch_end(self):
        if self.shuffle:
            self.inds_batch=[]
            for fold, (train_index, test_index) in enumerate(self.kf.split(self.selected_indices,self.selected_clst)):
                ids=[self.selected_indices[i] for i in test_index]
                self.inds_batch.append(ids)

    def __getitem__(self, index):
        code1 = self.X['code1'][self.inds_batch[index]]
        code2 = self.X['code2'][self.inds_batch[index]]
        code3 = self.X['code3'][self.inds_batch[index]]

        age = self.X['age'][self.inds_batch[index]]
        multi = self.X['multi'][self.inds_batch[index]]
        typ = self.X['typ'][self.inds_batch[index]]

        position = self.X['position'][self.inds_batch[index]]
        mask = self.X['mask'][self.inds_batch[index]]
        label = self.X['label'][self.inds_batch[index]]

        event = self.X['event'][self.inds_batch[index]]
        time = self.X['time'][self.inds_batch[index]]
        cluster = self.X['cluster'][self.inds_batch[index]]

        y = np.column_stack((time, event, cluster))
        if self.pretrain:
            return (code1,code2,code3,age,multi,typ,position,mask),{'model_only_mlm_head': label, 'classifier': cluster}
        else:
            return ((code1,code2,code3,age,multi,typ,position,mask),y),{"output_1": label, "output_4": y, "output_5": y}
    def __len__(self):
        return self.len_data











