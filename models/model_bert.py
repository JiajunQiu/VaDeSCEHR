import tensorflow as tf
import numpy as np


ACT2FN = {"gelu": tf.keras.activations.gelu, "relu": tf.keras.activations.relu, "tanh": tf.keras.activations.tanh}

class CustomSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='custom_sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Mask the labels that are -1
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        y_true = tf.reshape(y_true, [-1,])
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Call the parent class's update_state method with the masked values
        super().update_state(y_true, y_pred, sample_weight)


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    y_true = tf.reshape(y_true, [-1,])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True,ignore_class=-1)
    return tf.reduce_mean(loss)
'''

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    y_true = tf.reshape(y_true, [-1,])
    mask = tf.not_equal(y_true, -1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True,ignore_class=-1)
    return tf.reduce_mean(loss)
'''

class PositionalEmbeddings(tf.keras.layers.Embedding):
    def __init__(self, max_position_embeddings, hidden_size, **kwargs):
        super(PositionalEmbeddings, self).__init__(
            input_dim=max_position_embeddings,
            output_dim=hidden_size,
            weights=[self._init_posi_embedding(max_position_embeddings, hidden_size)],
            **kwargs
        )

    def _init_posi_embedding(self, max_position_embeddings=None, hidden_size=None):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embeddings, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embeddings):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embeddings):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return tf.convert_to_tensor(lookup_table, dtype=tf.float32)


class ModelEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config):
        super(ModelEmbeddings, self).__init__()

        self.word_embeddings1 = tf.keras.layers.Embedding(config['vocab_size1'], config['hidden_size'])
        self.word_embeddings1.build(input_shape=(None, config['vocab_size1']))
        self.word_embeddings2 = tf.keras.layers.Embedding(config['vocab_size2'], config['hidden_size'])
        self.word_embeddings2.build(input_shape=(None, config['vocab_size2']))
        self.word_embeddings3 = tf.keras.layers.Embedding(config['vocab_size3'], config['hidden_size'])
        self.word_embeddings3.build(input_shape=(None, config['vocab_size3']))

        self.age_embeddings = tf.keras.layers.Embedding(config['age_vocab_size'], config['hidden_size'])
        self.age_embeddings.build(input_shape=(None, config['age_vocab_size']))
        self.type_embeddings = tf.keras.layers.Embedding(config['type_vocab_size'], config['hidden_size'])
        self.type_embeddings.build(input_shape=(None, config['type_vocab_size']))
        self.posi_embeddings = PositionalEmbeddings(config['seq_length'], config['hidden_size'])

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, axis=-1)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, word_ids1, word_ids2, word_ids3, age_ids=None, type_ids=None, posi_ids=None):
        word_embed1 = self.word_embeddings1(word_ids1)
        word_embed2 = self.word_embeddings2(word_ids2)
        word_embed3 = self.word_embeddings3(word_ids3)

        mask1 = tf.cast(tf.equal(word_ids1, 1), tf.float32)
        mask2 = tf.cast(tf.equal(word_ids2, 1), tf.float32)
        mask3 = tf.cast(tf.equal(word_ids3, 1), tf.float32)


        word_embed1 = word_embed1*(1-mask1[...,tf.newaxis])+word_embed3*mask1[...,tf.newaxis]

        word_embed2 = word_embed2*(1-mask2[...,tf.newaxis])+word_embed1*mask2[...,tf.newaxis]
 
        word_embed3 = word_embed3*(1-mask3[...,tf.newaxis])+word_embed2*mask3[...,tf.newaxis]

        
        age_embed = self.age_embeddings(age_ids)
        type_embed = self.type_embeddings(type_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        embeddings_part = age_embed + type_embed +posi_embeddings
        embeddings = word_embed1 + word_embed2 + word_embed3 + embeddings_part
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings,embeddings_part

class ModelSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ModelSelfAttention, self).__init__(**kwargs)
        if config['hidden_size'] % config['num_attention_heads'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config['hidden_size'], config['num_attention_heads']))
        
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
        self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=False)

        self.dropout = tf.keras.layers.Dropout(config['attention_probs_dropout_prob'])
    def transpose_for_scores(self, x):
        x = tf.reshape(x,shape=[tf.shape(x)[0], -1, self.num_attention_heads, self.attention_head_size])
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x


    def call(self, inputs, attention_mask):
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
        attention_scores = attention_scores + attention_mask

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        attention_probs = tf.matmul(attention_probs, value_layer)
        attention_probs = tf.transpose(attention_probs, perm=[0, 2, 1, 3])
        output_shape = [batch_size, from_seq_len, self.num_attention_heads * self.attention_head_size]
        attention_probs = tf.reshape(attention_probs, output_shape)
        return attention_probs   
    
class ModelSelfOutput(tf.keras.Model):
    def __init__(self, config):
        super(ModelSelfOutput, self).__init__()
        self.dense = tf.keras.layers.Dense(config['hidden_size'])
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.dropout = tf.keras.layers.Dropout(config['hidden_dropout_prob'])

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
    
class ModelAttention(tf.keras.Model):
    def __init__(self, config):
        super(ModelAttention, self).__init__()
        self.self_attention = ModelSelfAttention(config)
        self.self_output = ModelSelfOutput(config)

    def call(self, input_tensor, attention_mask):
        self_output= self.self_attention(input_tensor, attention_mask)
        attention_output = self.self_output(self_output, input_tensor)
        return attention_output
    
class ModelIntermediate(tf.keras.Model):
    def __init__(self, config):
        super(ModelIntermediate, self).__init__()
        self.dense = tf.keras.layers.Dense(config['intermediate_size'])
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class ModelOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ModelOutput, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config['hidden_size'])
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12,axis=-1)
        self.dropout = tf.keras.layers.Dropout(config['hidden_dropout_prob'])

    def call(self,hidden_states, inputs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + inputs)
        return hidden_states


class ModelLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(ModelLayer, self).__init__()
        self.attention = ModelAttention(config)
        self.intermediate = ModelIntermediate(config)
        self.output_layer = ModelOutput(config)

    def call(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output_layer(intermediate_output, attention_output)
        return layer_output
    
class ModelEncoder(tf.keras.Model):
    def __init__(self, config):
        super(ModelEncoder, self).__init__()
        self.layer = [ModelLayer(config) for _ in range(config['num_hidden_layers'])]

    def call(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states= layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class ModelPooler(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(ModelPooler, self).__init__()
        self.dense = tf.keras.layers.Dense(hidden_size)
        self.activation = tf.keras.activations.tanh

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ModelPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(ModelPredictionHeadTransform, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config['hidden_size'])
        if isinstance(config['hidden_act'], str):
            self.transform_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.transform_act_fn = config['hidden_act']
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
    

class ModelLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, Model_model_embedding_weights, **kwargs):
        super(ModelLMPredictionHead, self).__init__(**kwargs)
        self.transform = ModelPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = tf.keras.layers.Dense(Model_model_embedding_weights.shape[0])
        self.decoder.build(input_shape=(None, Model_model_embedding_weights.shape[1]))
        self.decoder.weights[0].assign(tf.transpose(Model_model_embedding_weights))
#        self.bias = tf.Variable(tf.zeros(Model_model_embedding_weights.shape[0]),dtype=tf.float32)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class ModelOnlyMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, Model_model_embedding_weights):
        super(ModelOnlyMLMHead, self).__init__()
        self.predictions = ModelLMPredictionHead(config, Model_model_embedding_weights)

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    

class ModelATT(tf.keras.Model):
    def __init__(self, config):
        super(ModelATT, self).__init__(config)
        self.seq_length = config['seq_length']
        self.embeddings = ModelEmbeddings(config=config)
        self.encoder = ModelEncoder(config=config)
        self.output_all_encoded_layers = False
        self.attention_pool = tf.keras.layers.Dense(1)


    def call(self, inputs):
        input_ids1,input_ids2,input_ids3,age_ids,_,type_ids,posi_ids,attention_mask,=inputs
        if attention_mask is None:
            attention_mask = tf.ones_like(input_ids1)

        if age_ids is None:
            age_ids = tf.zeros_like(input_ids1)
        if type_ids is None:
            type_ids = tf.zeros_like(input_ids1)
        if posi_ids is None:
            posi_ids = tf.zeros_like(input_ids1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output,embedding_output_part = self.embeddings(input_ids1, input_ids2, input_ids3, age_ids, type_ids, posi_ids)

        encoded_layers = self.encoder(embedding_output,extended_attention_mask, output_all_encoded_layers=self.output_all_encoded_layers)

        sequence_output = encoded_layers[-1]

        extended_attention_mask = tf.squeeze(extended_attention_mask, [1])
        extended_attention_mask = tf.transpose(extended_attention_mask, [0, 2, 1])

        pooled_output = tf.squeeze(tf.matmul(tf.transpose(tf.nn.softmax(self.attention_pool(sequence_output)+extended_attention_mask, axis=1), perm=[0,2,1]), sequence_output),axis=-2)

        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, embedding_output, embedding_output_part

class ModelForMaskedLM(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(ModelForMaskedLM, self).__init__(**kwargs)
        self.ModelATT = ModelATT(config)
        self.cls = ModelOnlyMLMHead(config, self.ModelATT.embeddings.word_embeddings3.weights[0])

    def call(self, inputs):
        sequence_output, _ , _ , _ = self.ModelATT(inputs)
        prediction_scores = self.cls(sequence_output)
        return prediction_scores 