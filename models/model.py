import tensorflow as tf
import tensorflow_probability as tfp
import os

from models.networks import VAE_Encoder,Decoder_reshape
from models.model_bert import ModelForMaskedLM,CustomSparseCategoricalAccuracy,masked_sparse_categorical_crossentropy
import tensorflow_addons as tfa
from utils.utils import weibull_scale, weibull_log_pdf, tensor_slice

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class GMM_Survival(tf.keras.Model):
    def __init__(self, config):
        super(GMM_Survival, self).__init__(name="GMM_Survival")
        self.encoded_size = config['training']['latent_dim']
        self.num_clusters = config['training']['num_clusters']
        self.survival = config['training']['survival']
        self.s = config['training']['monte_carlo']
        self.sample_surv = config['training']['sample_surv']
        self.learn_prior = config['training']['learn_prior']
 
        self.encoder = ModelForMaskedLM(config['pre_training'])
        if os.path.isdir(config['training']['path']):
            inputs = [(None, 200), (None, 200),(None, 200), (None, 200),(None, 200), (None, 200),(None, 200), (None, 200)]
            self.encoder.build(inputs)
            pre_trained_model = tf.keras.models.load_model(config['training']['path'], custom_objects={'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy,'CustomSparseCategoricalAccuracy':CustomSparseCategoricalAccuracy()})
            self.encoder.set_weights(pre_trained_model.get_weights())
        else:
            self.encoder.load_weights(config['training']['path'])

        self.vae_encoder = VAE_Encoder(self.encoded_size)

        self.decoder_reshape = Decoder_reshape(config)

        self.c_mu = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size]), name='mu')
        self.log_c_sigma = tf.Variable(tf.initializers.GlorotNormal()([self.num_clusters, self.encoded_size]), name="sigma")
        # Cluster-specific survival model parameters
        self.c_beta = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size + 1]),
                                  name='beta')
        # Weibull distribution shape parameter
        self.weibull_shape = config['training']['weibull_shape']
        if self.learn_prior:
            self.prior_logits = tf.Variable(tf.ones([self.num_clusters]), name="prior")
        else:
            self.prior = tf.constant(tf.ones([self.num_clusters]) * (1 / self.num_clusters))
        self.use_t = tf.Variable([1.0], trainable=False)

    def call(self, inputs, training=True):
        # NB: inputs have to include predictors/covariates/features (x), time-to-event (t), and
        # event indicators (d). d[i] == 1 if the i-th event is a death, and d[i] == 0 otherwise.
        x, y = inputs
        t = y[:, 0]
        d = y[:, 1]
        _ , x , _ , embedding_output_part = self.encoder.ModelATT(x)
        z_mu, log_z_sigma = self.vae_encoder(x)
        tf.debugging.check_numerics(z_mu, message="z_mu")
        z = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        if training:
            z_sample = z.sample(self.s)
        else:
            z_sample = tf.expand_dims(z_mu, 0)

        tf.debugging.check_numerics(self.c_mu, message="c_mu")
        tf.debugging.check_numerics(self.log_c_sigma, message="c_sigma")
        c_sigma = tf.math.exp(self.log_c_sigma)

        # p(z|c)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=tf.cast(self.c_mu[i, :], tf.float64),
                                       scale_diag=tf.math.sqrt(tf.cast(c_sigma[i, :], tf.float64))).prob(
                tf.cast(z_sample, tf.float64)) + 1e-60) for i in range(self.num_clusters)], axis=-1)
        tf.debugging.check_numerics(p_z_c, message="p_z_c")

        # prior p(c)
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + 1e-60)
        else:
            prior = self.prior
        tf.debugging.check_numerics(prior, message="prior")

        if self.survival:
            # Compute Weibull distribution's scale parameter, given z and c
            tf.debugging.check_numerics(self.c_beta, message="c_beta")
            if self.sample_surv:
                lambda_z_c = tf.stack([weibull_scale(x=z_sample, beta=self.c_beta[i, :])
                                        for i in range(self.num_clusters)], axis=-1)
            else:
                lambda_z_c = tf.stack([weibull_scale(x=tf.stack([z_mu for i in range(self.s)], axis=0),
                                                     beta=self.c_beta[i, :]) for i in range(self.num_clusters)], axis=-1)
            tf.debugging.check_numerics(lambda_z_c, message="lambda_z_c")

            # Evaluate p(t|z,c), assuming t|z,c ~ Weibull(lambda_z_c, self.weibull_shape)
            p_t_z_c = tf.stack([weibull_log_pdf(t=t, d=d, lmbd=lambda_z_c[:, :, i], k=self.weibull_shape)
                                for i in range(self.num_clusters)], axis=-1)
            p_t_z_c = tf.clip_by_value(p_t_z_c, -1e+64, 1e+64)
            tf.debugging.check_numerics(p_t_z_c, message="p_t_z_c")
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64) + p_t_z_c
        else:
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)

        p_c_z = tf.nn.log_softmax(p_c_z, axis=-1)
        p_c_z = tf.math.exp(p_c_z)
        tf.debugging.check_numerics(p_c_z, message="p_c_z")

        if self.survival:
            loss_survival = -tf.reduce_sum(tf.multiply(p_t_z_c, tf.cast(p_c_z, tf.float64)), axis=-1)
            tf.debugging.check_numerics(loss_survival, message="loss_survival")

        loss_clustering = - tf.reduce_sum(tf.multiply(tf.cast(p_c_z, tf.float64), tf.cast(p_z_c, tf.float64)),
                                                  axis=-1)

        loss_prior = - tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64), 1e-60 +
                                                                tf.cast(prior, tf.float64)), axis=-1)

        loss_variational_1 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        loss_variational_2 = tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64),
                                                                      1e-60 + tf.cast(p_c_z, tf.float64)), axis=-1)

        tf.debugging.check_numerics(loss_clustering, message="loss_clustering")
        tf.debugging.check_numerics(loss_prior, message="loss_prior")
        tf.debugging.check_numerics(loss_variational_1, message="loss_variational_1")
        tf.debugging.check_numerics(loss_variational_2, message="loss_variational_2")

        if self.survival:
            self.add_loss(tf.math.reduce_mean(loss_survival))
        self.add_loss(tf.math.reduce_mean(loss_clustering))
        self.add_loss(tf.math.reduce_mean(loss_prior))
        self.add_loss(tf.math.reduce_mean(loss_variational_1))
        self.add_loss(tf.math.reduce_mean(loss_variational_2))

        # Logging metrics in TensorBoard
        self.add_metric(loss_clustering, name='loss_clustering', aggregation="mean")
        self.add_metric(loss_prior, name='loss_prior', aggregation="mean")
        self.add_metric(loss_variational_1, name='loss_variational_1', aggregation="mean")
        self.add_metric(loss_variational_2, name='loss_variational_2', aggregation="mean")
        if self.survival:
            self.add_metric(loss_survival, name='loss_survival', aggregation="mean")


        extended_attention_mask = tf.expand_dims(tf.expand_dims(inputs[0][-1], 1), 2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        dec = self.decoder_reshape(tf.reshape(z_sample,(-1,z_sample.shape[-1])))+embedding_output_part
        dec = self.encoder.cls(self.encoder.ModelATT.encoder(dec,extended_attention_mask,output_all_encoded_layers=False)[-1])

        # Evaluate risk scores based on hard clustering assignments
        # Survival time may ba unobserved, so a special procedure is needed when time is not observed...
        p_z_c = p_z_c[0]    # take the first sample
        p_c_z = p_c_z[0]
     
        if self.survival:
            lambda_z_c = lambda_z_c[0]  # Take the first sample
            # Use Bayes rule to compute p(c|z) instead of p(c|z,t), since t is unknown
            p_c_z_nt = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)
            p_c_z_nt = tf.nn.log_softmax(p_c_z_nt, axis=-1)
            p_c_z_nt = tf.math.exp(p_c_z_nt)
            inds_nt = tf.dtypes.cast(tf.argmax(p_c_z_nt, axis=-1), tf.int32)
            risk_scores_nt = tensor_slice(target_tensor=tf.cast(lambda_z_c, tf.float64), index_tensor=inds_nt)

            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risk_scores_t = tensor_slice(target_tensor=lambda_z_c, index_tensor=inds)

            p_c_z = tf.cond(self.use_t[0] < 0.5, lambda: p_c_z_nt, lambda: p_c_z)
            risk_scores = tf.cond(self.use_t[0] < 0.5, lambda: risk_scores_nt, lambda: risk_scores_t)
        else:
            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risk_scores = tensor_slice(target_tensor=p_c_z, index_tensor=inds)
            lambda_z_c = risk_scores

        p_z_c = tf.cast(p_z_c, tf.float64)

        dec = tf.reshape(dec,(-1,self.s,dec.shape[-2],dec.shape[-1]))
        z_sample = tf.transpose(z_sample, [1, 0, 2])
        risk_scores = tf.expand_dims(risk_scores, -1)
        return dec, z_sample, p_z_c, p_c_z, risk_scores, lambda_z_c, y

    def generate_samples(self, j, n_samples):
        z = tfd.MultivariateNormalDiag(loc=self.c_mu[j, :], scale_diag=tf.math.sqrt(tf.math.exp(self.log_c_sigma[j, :])))
        z_sample = z.sample(n_samples)
        dec = self.decoder(tf.expand_dims(z_sample, 0))
        return dec
