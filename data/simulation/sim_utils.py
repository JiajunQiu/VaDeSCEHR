"""
Utility functions for numerical simulations.
"""
import numpy as np

from sklearn.datasets import make_low_rank_matrix

import pandas as pd

import tensorflow as tf

import numpy as np
from numpy.random import multivariate_normal, uniform, choice

from sklearn.datasets import make_spd_matrix

from scipy.stats import weibull_min

from sklearn.datasets import make_low_rank_matrix


def simulate_nonlin_profile_surv(icd_codes, n: int, k: int, max_pos: int, hidden_size: int, num_layer:int, num_head: int ,latent_dim: int, p_cens: float, seed: int, p_c=None,
                                 balanced=True, clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                                 clust_intercepts=True, weibull_k=1, xrange=[-5, 5], brange=[-1, 1]):
    """
    Simulates data with heterogeneous survival profiles and nonlinear (!) relationships
    (covariates are generated from latent features using an MLP decoder).
    """
    # Replicability
    np.random.seed(seed)
    p=len(icd_codes)
    # Sanity checks
    assert p > 0 and latent_dim > 0 and n > 0 and k > 0
    assert 1 < k < n
    assert latent_dim < p
    assert len(xrange) == 2 and xrange[0] < xrange[1]
    assert len(brange) == 2 and brange[0] < brange[1]
    assert weibull_k > 0

    # Cluster prior prob-s
    if p_c is not None:
        assert len(p_c) == k and sum(p_c) == 1
    else:
        if balanced:
            p_c = np.ones((k, )) / k
        else:
            p_c = uniform(0, 1, (k, ))
            p_c = p_c / np.sum(p_c)

    # Cluster assignments
    c = choice(a=np.arange(k), size=(n, ), replace=True, p=p_c)

    # Cluster-specific means
    means = np.zeros((k, latent_dim))
    mu = uniform(xrange[0], xrange[1], (1, latent_dim))
    for l in range(k):
        if clust_mean:
            mu_l = uniform(xrange[0], xrange[1], (1, latent_dim))
            means[l, :] = mu_l
        else:
            means[l, :] = mu

    # Cluster-specific covariances
    cov_mats = []
    sigma = make_spd_matrix(latent_dim, random_state=seed)
    if isotropic:
        sigma = sigma * np.eye(latent_dim)
    for l in range(k):
        if clust_cov:
            sigma_l = make_spd_matrix(latent_dim, random_state=(seed + l))
            if isotropic:
                sigma_l = sigma_l * np.eye(latent_dim)
            cov_mats.append(sigma_l)
        else:
            cov_mats.append(sigma)

    # Latent features
    Z = np.zeros((n, latent_dim))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = multivariate_normal(mean=means[l, :], cov=cov_mats[l], size=n_l)
        Z[c == l, :] = Z_l

    # Predictors
    '''
    mlp_dec1 = random_nonlin_map(n_in=latent_dim, n_out=p*max_pos)
    X = mlp_dec1(Z)
    X = X.reshape(n, max_pos, p)
    '''
    mlp_dec1 = random_nonlin_map(n_in=latent_dim, n_out=hidden_size*max_pos)
    X = mlp_dec1(Z)
    X = X.reshape(n, max_pos, hidden_size)

    # Cluster-specific coefficients for the survival model
    coeffs = np.zeros((k, latent_dim))
    intercepts = np.zeros((k, ))
    beta = uniform(brange[0], brange[1], (1, latent_dim))
    beta0 = uniform(brange[0], brange[1], (1, 1))
    for l in range(k):
        if clust_coeffs:
            beta_l = uniform(brange[0], brange[1], (1, latent_dim))
            coeffs[l, :] = beta_l
        else:
            coeffs[l, :] = beta
        if clust_intercepts:
            beta0_l = uniform(brange[0], brange[1], (1, 1))
            intercepts[l] = beta0_l
        else:
            intercepts[l] = beta0

    # Survival times
    t = np.zeros((n, ))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = Z[c == l, :]
        coeffs_l = np.expand_dims(coeffs[l, :], 1)
        intercept_l = intercepts[l]
        logexps_l = np.log(1 + np.exp(intercept_l + np.squeeze(np.matmul(Z_l, coeffs_l))))

        t_l = weibull_min.rvs(weibull_k, loc=0, scale=logexps_l, size=n_l)

        t[c == l] = t_l

    # Censoring
    # NB: d == 1 if failure; 0 if censored
    d = (uniform(0, 1, (n, )) >= p_cens) * 1.0
    t_cens = uniform(0, t, (n, ))
    t[d == 0] = t_cens[d == 0]

    return X, t, d, c, Z, means, cov_mats, coeffs, intercepts


class pseudo_att(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, num_heads=8,rank=1000):
        super(pseudo_att, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be multiple of num_heads"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads    

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=False)

        W_0 = make_low_rank_matrix(self.input_size, self.all_head_size, effective_rank=rank)
        W_1 = make_low_rank_matrix(self.input_size, self.all_head_size, effective_rank=rank)
        W_2 = make_low_rank_matrix(self.input_size, self.all_head_size, effective_rank=rank)

        self.query.kernel = tf.Variable(W_0, dtype=tf.float64)
        self.key.kernel = tf.Variable(W_1, dtype=tf.float64)
        self.value.kernel = tf.Variable(W_2, dtype=tf.float64)

    def transpose_for_scores(self, x):
        x = tf.reshape(x,shape=[tf.shape(x)[0], -1, self.num_attention_heads, self.attention_head_size])
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x


    def call(self, inputs,mask):
        input_shape  = tf.shape(inputs)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]

        hidden_states = inputs
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=-1)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.attention_head_size, tf.float32))

        attention_scores = attention_scores+mask

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        attention_probs = tf.matmul(attention_probs, value_layer)
        attention_probs = tf.transpose(attention_probs, perm=[0, 2, 1, 3])
        output_shape = [batch_size, from_seq_len, self.num_attention_heads * self.attention_head_size]
        attention_probs = tf.reshape(attention_probs, output_shape)

        return attention_probs


def random_nonlin_map(n_in, n_out, rank=1000):
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_out, effective_rank=rank)
    b_0 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: ReLU(np.matmul(x, W_0))+ np.tile(b_0, (x.shape[0], 1))


    return nlin_map




def random_nonlin_map_basic(n_in, n_out, n_hidden, rank=1000):
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # Disabled biases for now...
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))),
                                                       W_1) + np.tile(b_1, (x.shape[0], 1))), W_2) + \
                         np.tile(b_2, (x.shape[0], 1))

    return nlin_map

def ReLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pp(start, end, n):
    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.DatetimeIndex((10**9*np.random.randint(start_u, end_u, n, dtype=np.int64)).view('M8[ns]'))
