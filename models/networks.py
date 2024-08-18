"""
Encoder and decoder architectures used by VaDeSC.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

'''
# Wide MLP encoder and decoder architectures
class VAE_Encoder_uccd(layers.Layer):
    def __init__(self, encoded_size):
        super(VAE_Encoder_uccd, self).__init__(name='VAE_Encoder_uccd')
        self.dense_layers1 = [tfkl.Dense(50, activation='relu') for x in range(2)]
        self.dense_layers2 = [tfkl.Dense(50, activation='relu') for x in range(2)]
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):

        mu=inputs
        sigma=inputs
        for layer in self.dense_layers1:
            mu = layer(mu)

        for layer in self.dense_layers2:
            sigma = layer(sigma)

        mu = self.mu(inputs)
        sigma = self.sigma(inputs)


        return mu, sigma
'''

# Wide MLP encoder and decoder architectures
class VAE_Encoder_uccd(layers.Layer):
    def __init__(self, encoded_size):
        super(VAE_Encoder_uccd, self).__init__(name='VAE_Encoder_uccd')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(int((encoded_size*(encoded_size+1))/2), activation=None)

    def call(self, inputs, **kwargs):
        mu = self.mu(inputs)
        sigma = self.sigma(inputs)
        return mu, sigma

# Wide MLP encoder and decoder architectures
class VAE_Encoder(layers.Layer):
    def __init__(self, encoded_size):
        super(VAE_Encoder, self).__init__(name='vae_encoder')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        mu = self.mu(inputs)
        sigma = self.sigma(inputs)
        return mu, sigma

class Decoder_reshape(layers.Layer):
    def __init__(self, config):
        super(Decoder_reshape, self).__init__(name='dec_rs')
        self.dense1 = tfkl.Dense(256, activation='relu')
        self.dense_layers = [tfkl.Dense(2**(8+(x+1)), activation='relu') for x in range(config['training']['num_reshape_layers']-1)]
        self.dense5 = tfkl.Dense(config['pre_training']['seq_length']*config['pre_training']['hidden_size'], activation=None)
        self.dense6 = tfkl.Reshape((config['pre_training']['seq_length'], config['pre_training']['hidden_size']))
        self.dropout = tf.keras.layers.Dropout(config['training']['dropout_prob'])
    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dropout(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x



# Wide MLP encoder and decoder architectures
class Encoder(layers.Layer):
    def __init__(self, encoded_size,config):
        super(Encoder, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(config['training']['dropout_prob'])
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        x = tfkl.Flatten()(inputs)
#        x=inputs
        x = self.dense1(x)
        x = self.dropout(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

class CLS(layers.Layer):
    def __init__(self, input_shape, activation=None):
        super(CLS, self).__init__(name='CLS')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(2000, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(500, activation='relu')
        if activation == "sigmoid":
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, input_shape, activation=None):
        super(Decoder, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(2000, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(500, activation='relu')
        if activation == "sigmoid":
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


# VGG-based architectures
class VGGConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGConvBlock, self).__init__(name="VGGConvBlock{}".format(block_id))
        self.conv1 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.maxpool = tfkl.MaxPooling2D((2, 2))

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out


class VGGDeConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGDeConvBlock, self).__init__(name="VGGDeConvBlock{}".format(block_id))
        self.upsample = tfkl.UpSampling2D((2, 2), interpolation='bilinear')
        self.convT1 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')

    def call(self, inputs, **kwargs):
        out = self.upsample(inputs)
        out = self.convT1(out)
        out = self.convT2(out)

        return out


class VGGEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(VGGEncoder, self).__init__(name='VGGEncoder')
        self.layers = [VGGConvBlock(32, 1), VGGConvBlock(64, 2)]
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        out = inputs

        # Iterate through blocks
        for block in self.layers:
            out = block(out)
        out_flat = tfkl.Flatten()(out)
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma


class VGGDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(VGGDecoder, self).__init__(name='VGGDecoder')

        target_shape = (13, 13, 64)     # 64 x 64

        self.activation = activation
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.layers = [VGGDeConvBlock(64, 1), VGGDeConvBlock(32, 2)]
        self.convT = tfkl.Conv2DTranspose(filters=input_shape[2], kernel_size=3, padding='same')

    def call(self, inputs, **kwargs):
        out = self.dense(inputs[0])
        out = self.reshape(out)

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        # Last convolution
        out = self.convT(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return tf.expand_dims(out, 0)


# Smaller encoder and decoder architectures for low-dimensional datasets
class Encoder_small(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder_small, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(50, activation='relu')
        self.dense2 = tfkl.Dense(100, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        x = tfkl.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder_small(layers.Layer):
    def __init__(self, input_shape, activation):
        super(Decoder_small, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(100, activation='relu')
        self.dense2 = tfkl.Dense(50, activation='relu')
        if activation == "sigmoid":
            print("yeah")
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense4(x)
        return x
