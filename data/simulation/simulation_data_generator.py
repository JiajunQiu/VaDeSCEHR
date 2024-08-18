from sim_utils import simulate_nonlin_profile_surv
import numpy as np
import pickle
from sim_utils import random_nonlin_map, pseudo_att
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


corpus=['UNK']
for l in open('../corpus_code2.txt'):
    l=l.rstrip()
    if l not in corpus:
        corpus.append(l)

scaler = StandardScaler()
max_pos=100
n=30000
num_layer=3
seed=42
np.random.seed(seed)
p=len(corpus)
hidden_size=100
num_head=10

sample_step=100

X, t, d, c, Z, mus, sigmas, betas, betas_0  = simulate_nonlin_profile_surv(corpus, n=n, k=3,max_pos=max_pos,hidden_size=hidden_size,num_layer=num_layer, num_head=num_head, p_cens=0.3, seed=42, latent_dim=5,clust_mean=True, clust_cov=True,isotropic=True,
                                          clust_intercepts=False, weibull_k=1, xrange=[-10, 10],brange=[-2.5, 2.5])

att_layers=[]
rank=1000
for _ in range(num_layer):
    att = pseudo_att(hidden_size,hidden_size,num_head,rank=rank)
    att_layers.append(att)


icd_codes=np.array(corpus)
X_label = []
X_=[]
for i in range(0, n, sample_step):
    X_samples = X[i:i+sample_step]
    X_att=X_samples
    mask=np.ones((sample_step,max_pos))
    for idx,pos in enumerate(np.random.choice(range(5,max_pos), sample_step)):
        mask[idx,pos:]=0

    X_samples = X_samples * mask[:, :, np.newaxis]  # Element-wise multiplication between X and mask
    
    extended_attention_mask = tf.expand_dims(tf.expand_dims(mask, 1), 2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.float32)
    
    for att in att_layers:
        X_att=att.call(X_att,extended_attention_mask)+X_samples

    X_att = X_att*mask[:, :, np.newaxis]
    X_.extend([xxx for xxx in X_att])


    mlp_dec2 = random_nonlin_map(n_in=hidden_size, n_out=p)
    X_samples = tf.map_fn(mlp_dec2, X_att, dtype=tf.float32)

    exp_X_samples = np.exp(X_samples)

    softmax_X_samples = exp_X_samples / np.sum(exp_X_samples, axis=-1, keepdims=True)

    class_idx = np.argmax(softmax_X_samples, axis=-1)

    for idx,i in enumerate(class_idx):
        tmp=icd_codes[i]
        tmp[mask[idx]==0]='PAD'
        X_label.append(list(tmp))

x = np.array(X_label)
#x = np.array(X_)
num_fold=3
# Renaming

e = d

print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
print(x[0].shape)
idx = np.arange(0, x.shape[0])
print("x_shape:{}".format(x.shape))

np.random.shuffle(idx)
x = x[idx]
t = t[idx]
e = e[idx]
c = c[idx]
end_time = max(t)
print("end_time:{}".format(end_time))
print("observed percent:{}".format(sum(e) / len(e)))

sublist_size = len(idx) // num_fold
print('sublist_size',sublist_size)
sublists = [idx[i:i+sublist_size] for i in range(0, len(idx), sublist_size)]

data=[]
for fold_idx in sublists:
    tmp={'input':x[fold_idx],'time':t[fold_idx],'event':e[fold_idx],'cluster':c[fold_idx]}
    data.append(tmp)

out=open('benchmark_data.pkl','wb')
pickle.dump(data,out)
out.close()

