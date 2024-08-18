"""
Utility functions for numerical simulations.
"""
import numpy as np

from sklearn.datasets import make_low_rank_matrix

import pandas as pd

import tensorflow as tf

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
