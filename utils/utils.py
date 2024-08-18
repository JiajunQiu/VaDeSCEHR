"""
miscellaneous utility functions.
"""
import matplotlib

import matplotlib.pyplot as plt
import logging
from scipy.optimize import linear_sum_assignment as linear_assignment

import numpy as np

from scipy.stats import weibull_min, fisk

import sys


import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


sys.path.insert(0, '../../')


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred.size
  

def sample_weibull(scales, shape, n_samples=200):
    return np.transpose(weibull_min.rvs(shape, loc=0, scale=scales, size=(n_samples, scales.shape[0])))



# Weibull(lmbd, k) log-pdf
def weibull_log_pdf(t, d, lmbd, k):
    t_ = tf.ones_like(lmbd) * tf.cast(t, tf.float64)
    d_ = tf.ones_like(lmbd) * tf.cast(d, tf.float64)
    k = tf.cast(k, tf.float64)
    a = t_ / (1e-60 + tf.cast(lmbd, tf.float64))
    tf.debugging.check_numerics(a, message="weibull_log_pdf")

    return tf.cast(d_, tf.float64) * (tf.math.log(1e-60 + k) - tf.math.log(1e-60 + tf.cast(lmbd, tf.float64)) +
                                     (k - 1) * tf.math.log(1e-60 + tf.cast(t_, tf.float64)) - (k - 1) *
                                     tf.math.log(1e-60 + tf.cast(lmbd, tf.float64))) - (a) ** k


def weibull_scale(x, beta):
    beta_ = tf.cast(beta, tf.float64)
    beta_ = tf.cast(tf.ones([tf.shape(x)[0], tf.shape(x)[1], beta.shape[0]]), tf.float64) * beta_
    return tf.clip_by_value(tf.math.log(1e-60 + 1.0 + tf.math.exp(tf.reduce_sum(-tf.cast(x, tf.float64) * beta_[:, :, :-1], axis=2) -
                                                 tf.cast(beta[-1], tf.float64))), -1e+64, 1e+64)


def sample_weibull_mixture(scales, shape, p_c, n_samples=200):
    scales_ = np.zeros((scales.shape[0], n_samples))
    cs = np.zeros((scales.shape[0], n_samples)).astype(int)
    for i in range(scales.shape[0]):
        cs[i] = np.random.choice(a=np.arange(0, p_c.shape[1]), p=p_c[i], size=(n_samples,))
        scales_[i] = scales[i, cs[i]]
    return scales_ * np.random.weibull(shape, size=(scales.shape[0], n_samples))


def tensor_slice(target_tensor, index_tensor):
    indices = tf.stack([tf.range(tf.shape(index_tensor)[0]), index_tensor], 1)
    return tf.gather_nd(target_tensor, indices)
