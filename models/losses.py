"""
Loss functions for the reconstruction term of the ELBO.
"""
import tensorflow as tf


class Losses:
    def __init__(self, configs):
        self.input_dim = configs['pre_training']['seq_length']
        self.tuple = False
        if isinstance(self.input_dim, list):
            print("\nData is tuple!\n")
            self.tuple = True
            self.input_dim = self.input_dim[0] * self.input_dim[1]

    def loss_reconstruction_binary(self, inp, x_decoded_mean):
        x = inp
        # NB: transpose to make the first dimension correspond to MC samples
        if self.tuple:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2, 3])
        else:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
        loss = self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean[i])
                                                              for i in range(x_decoded_mean.shape[0])], axis=-1),
                                                    axis=-1)
        return loss

    def loss_sparse_categorical_crossentropy(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = tf.reshape(y_true, [-1,])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True,ignore_class=-1)
        return self.input_dim * tf.reduce_mean(loss)
    '''
#    v2
    def loss_sparse_categorical_crossentropy(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = tf.reshape(y_true, [-1,])
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True,ignore_class=-1)
        return self.input_dim * tf.reduce_mean(loss)
    '''
    '''
    def loss_reconstruction_mse(self, inp, x_decoded_mean):
        # NB: transpose to make the first dimension correspond to MC samples
        x_decoded_mean = tf.transpose(x_decoded_mean, perm=[0, 2, 1, 3])

        loss = self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x_decoded_mean[0][i], x_decoded_mean[1][i])
                                                              for i in range(x_decoded_mean.shape[1])], axis=-1),
                                                    axis=-1)
        return loss
    '''
    def loss_reconstruction_mse(self, inp, x_decoded_mean):
        inp=x_decoded_mean[0]
        x_decoded_mean=x_decoded_mean[1]
        x=tf.reshape(inp, [inp.shape[0], inp.shape[1]* inp.shape[2]])
        x_decoded_mean=tf.reshape(x_decoded_mean, [x_decoded_mean.shape[0], x_decoded_mean.shape[1],x_decoded_mean.shape[2]* x_decoded_mean.shape[3]])
        # NB: transpose to make the first dimension correspond to MC samples
        if self.tuple:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2, 3])
        else:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
        loss = self.hidden_dim* self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                                              for i in range(x_decoded_mean.shape[0])], axis=-1),
                                                    axis=-1)
        return loss

    def loss_reconstruction_mse_sim(self, inp, x_decoded_mean):
        x=tf.reshape(inp, [inp.shape[0], inp.shape[1]* inp.shape[2]])
        x_decoded_mean=tf.reshape(x_decoded_mean, [x_decoded_mean.shape[0], x_decoded_mean.shape[1],x_decoded_mean.shape[2]* x_decoded_mean.shape[3]])
        # NB: transpose to make the first dimension correspond to MC samples
        if self.tuple:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2, 3])
        else:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
        loss = self.hidden_dim* self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                                              for i in range(x_decoded_mean.shape[0])], axis=-1),
                                                    axis=-1)
        return loss
    
    def BinaryCrossentropy(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = tf.reshape(y_true, [-1,])
        loss = tf.keras.losses.BinaryCrossentropy(y_true, y_pred)
        return  tf.reduce_mean(loss)