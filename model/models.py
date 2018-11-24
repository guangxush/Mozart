# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, \
    Reshape, ReLU, ZeroPadding1D
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import keras.backend as K

from layers import Attention, RecurrentAttention, RecurrentCNN, KMaxPooling, Folding
from utils import get_score_joint, get_score_aspect, get_score_senti, pad_sequences_2d


# TODO: refactor the code

# callback for aspect model
class AspectModelMetrics(Callback):
    def __init__(self):
        super(AspectModelMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s_macro = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict(self.validation_data[0])
        _val_f1_macro = get_score_aspect(self.validation_data[1], valid_results)
        logs['val_f1_macro'] = _val_f1_macro
        self.val_f1s_macro.append(_val_f1_macro)
        print('val_f1_macro: %f' % _val_f1_macro)
        return


# model for aspect classification
class AspectModel(object):
    def __init__(self, config, train_data, valid_data, test_text, test_label=None):
        self.config = config
        self.weights = np.load('data/%s_embeddings.npy' % self.config.level)
        self.n_class = self.config.n_aspect
        self.max_len = self.config.max_len[self.config.level]
        self.lstm_units = 128

        self.x_train, self.y_train = train_data
        self.y_train = np.array(self.y_train)
        self.x_valid, self.y_valid = valid_data
        self.y_valid = np.array(self.y_valid)
        self.x_test = test_text

        if test_label is not None:
            self.has_test_label = True
            self.y_test = np.array(test_label)
        else:
            self.has_test_label = False
            self.y_test = None

        self.callbacks = []
        self.init_callbacks()

        self.model = None
        if isinstance(self.config.aspect_model_type, list):
            self.build_merged_model(self.config.aspect_model_type)
        else:
            self.build_model(self.config.aspect_model_type)

    def pad_sequences(self, input_seq):
        if self.config.aspect_model_type == 'han':
            return pad_sequences_2d(input_seq, max_sents=self.config.han_max_sent,
                                    max_words=self.config.han_max_sent_len[self.config.level])
        else:
            return pad_sequences(input_seq, maxlen=self.max_len)

    def init_callbacks(self):
        self.callbacks.append(AspectModelMetrics())
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

    def build_base_network(self, aspect_model_type):
        if aspect_model_type == 'cnn':
            base_network = self.cnn()
        elif aspect_model_type == 'multicnn':
            base_network = self.multi_cnn()
        elif aspect_model_type == 'bilstm':
            base_network = self.bilstm()
        elif aspect_model_type == 'dpcnn':
            base_network = self.dpcnn()
        elif aspect_model_type == 'rnncnn':
            base_network = self.rnncnn()
        elif aspect_model_type == 'han':
            base_network = self.han()
        elif aspect_model_type == 'rcnn':
            base_network = self.rcnn()
        elif aspect_model_type == 'rcnn1':
            base_network = self.rcnn_1()
        elif aspect_model_type == 'vdcnn':
            base_network = self.vdcnn(self.config.vd_depth, self.config.vd_pooling_type, self.config.vd_use_shortcut)
        elif aspect_model_type == 'dcnn':
            base_network = self.dcnn()
        else:
            raise Exception('Aspect Model Type `%s` Not Understood' % aspect_model_type)
        return base_network

    def build_model(self, aspect_model_type):
        base_network = self.build_base_network(aspect_model_type)

        if aspect_model_type == 'han':
            input_text = Input(shape=(self.config.han_max_sent, self.config.han_max_sent_len[self.config.level]))
        else:
            input_text = Input(shape=(self.max_len,))
        vector_text = base_network(input_text)
        dense_layer = Dense(256, activation='relu')(vector_text)

        output_layer = Dense(self.n_class, activation='sigmoid')(dense_layer)

        # multi-label classification
        self.model = Model(input_text, output_layer)
        self.model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)

    def build_merged_model(self, aspect_model_types):
        if len(aspect_model_types) <= 1:
            raise ValueError('Models less than 2: `{}`'.format(aspect_model_types))

        models = []
        input_text = Input(shape=(self.max_len,))
        for aspect_model_type in aspect_model_types:
            if aspect_model_type == 'han':
                raise ValueError('han model is not supoorted!')
            base_network = self.build_base_network(aspect_model_type)

            vector_text = base_network(input_text)
            dense_layer = Dense(256, activation='relu')(vector_text)

            output_layer = Dense(self.n_class, activation='sigmoid')(dense_layer)
            models.append(output_layer)

        weighted_predict = self.weighted_merge(len(models), self.n_class)(concatenate(models, axis=-1))

        # multi-label classification
        self.model = Model(input_text, weighted_predict)
        self.model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)

    def train(self):
        x_train_pad = self.pad_sequences(self.x_train)
        x_valid_pad = self.pad_sequences(self.x_valid)

        print('training...')
        self.model.fit(x_train_pad, self.y_train, epochs=self.config.num_epochs, batch_size=self.config.batch_size,
                       validation_data=(x_valid_pad, self.y_valid), callbacks=self.callbacks)
        print('training end...')

        print('score over valid data:')
        valid_pred = self.model.predict(x_valid_pad)
        get_score_aspect(self.y_valid, valid_pred)

    def load(self):
        print("Loading model checkpoint {} ...\n".format('%s.hdf5' % self.config.exp_name))
        self.model.load_weights(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print("Model loaded")

    def score(self, input_data, label):
        input_data_pad = self.pad_sequences(input_data)
        pred = self.model.predict(input_data_pad)
        get_score_aspect(label, pred)

    def predict(self, input_data, threshold=0.5):
        input_data_pad = self.pad_sequences(input_data)
        y_pred = self.model.predict(input_data_pad)
        y_pred = np.array([[1 if y_pred[i, j] >= threshold else -2 for j in range(y_pred.shape[1])]
                           for i in range(y_pred.shape[0])])
        return y_pred

    def cnn(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                strides=1, activation='relu')(embedding_layer)
            maxpooling = MaxPool1D(pool_size=self.max_len - filter_length + 1)(conv_layer)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        concatenate_layer = concatenate(inputs=conv_layers)
        return Model(input_text, concatenate_layer)

    def multi_cnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer_1 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(embedding_layer)
            bn_layer_1 = BatchNormalization()(conv_layer_1)
            conv_layer_2 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(bn_layer_1)
            bn_layer_2 = BatchNormalization()(conv_layer_2)
            maxpooling = MaxPooling1D(pool_size=self.max_len - 2*filter_length + 2)(bn_layer_2)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        concatenate_layer = concatenate(conv_layers)
        return Model(input_text, concatenate_layer)

    def bilstm(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)
        bilstm_layer = Bidirectional(LSTM(128))(embedding_layer)
        dropout = Dropout(self.config.dropout)(bilstm_layer)

        return Model(input_text, dropout)

    def dpcnn(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        repeat = 3
        size = self.max_len
        region_x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(embedding_layer)
        x = Activation(activation='relu')(region_x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Add()([x, region_x])

        for _ in range(repeat):
            px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
            size = int((size + 1) / 2)
            x = Activation(activation='relu')(px)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Activation(activation='relu')(x)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Add()([x, px])

        x = MaxPooling1D(pool_size=size)(x)
        x = Flatten()(x)

        return Model(input_text, x)

    # end to end rnncnn
    def rnncnn(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        x = SpatialDropout1D(0.2)(embedding_layer)
        x = Bidirectional(GRU(100, return_sequences=True))(x)

        x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])

        return Model(input_text, conc)

    def han(self):
        def word_encoder():
            input_words = Input(shape=(self.config.han_max_sent_len[self.config.level],))
            word_vectors = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                                     weights=[self.weights], mask_zero=True)(input_words)
            sent_encoded = Bidirectional(GRU(self.lstm_units, return_sequences=True))(word_vectors)
            return Model(input_words, sent_encoded)

        def sentence_encoder():
            input_sents = Input(shape=(self.config.han_max_sent, self.lstm_units*2))
            sents_masked = Masking()(input_sents)   # support masking
            doc_encoded = Bidirectional(GRU(self.lstm_units, return_sequences=True))(sents_masked)
            return Model(input_sents, doc_encoded)

        input_text = Input(shape=(self.config.han_max_sent, self.config.han_max_sent_len[self.config.level]))
        sent_encoded = TimeDistributed(word_encoder())(input_text)  # word encoder
        sent_vectors = TimeDistributed(Attention(bias=True))(sent_encoded)  # word attention
        doc_encoded = sentence_encoder()(sent_vectors)  # sentence encoder
        doc_vector = Attention(bias=True)(doc_encoded)    # sentence attention

        return Model(input_text, doc_vector)

    # my implementation of recurrent convolution neural network
    # ValueError: An operation has `None` for gradient.
    # Please make sure that all of your ops have a gradient defined (i.e. are differentiable)
    def rcnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights],
                                    mask_zero=True)(input_text)
        x = SpatialDropout1D(0.2)(embedding_layer)
        x = RecurrentCNN(units=self.lstm_units)(x)

        x = TimeDistributed(Dense(self.lstm_units, activation='tanh'))(x)   # equation (4)
        max_pool = GlobalMaxPooling1D()(x)  # equation (5)

        return Model(input_text, max_pool)

    # implementation of recurrent convolution neural network I found from github (slightly modified)
    # see https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier
    def rcnn_1(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)

        doc_embedding = SpatialDropout1D(0.2)(embedding_layer)
        # We shift the document to the right to obtain the left-side contexts
        l_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                     x[:, :-1]], axis=1))(doc_embedding)
        # We shift the document to the left to obtain the right-side contexts
        r_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, 1:]], axis=1))(doc_embedding)

        # use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(self.lstm_units, return_sequences=True)(l_embedding)    # See equation (1)
        backward = LSTM(self.lstm_units, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2)
        # Keras returns the output sequences in reverse order.
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
        together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

        # use conv1D instead of TimeDistributed and Dense
        semantic = Conv1D(self.lstm_units, kernel_size=1, activation="tanh")(together)  # See equation (4).

        pool_rnn = Lambda(lambda x: K.max(x, axis=1))(semantic) # See equation (5).

        return Model(input_text, pool_rnn)

    # implementation of very deep convolution neural network
    # Reference: https://github.com/zonetrooper32/VDCNN/tree/tensorflow_version (modified)
    def vdcnn(self, depth, pooling_type, use_shortcut):
        def conv_block(inputs, filters, use_shortcut, shortcut):
            conv_1 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(inputs)
            bn_1 = BatchNormalization()(conv_1)
            relu_1 = ReLU()(bn_1)
            conv_2 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(relu_1)
            bn_2 = BatchNormalization()(conv_2)
            relu_2 = ReLU()(bn_2)

            if shortcut is not None and use_shortcut:
                return Add()([inputs, shortcut])
            else:
                return relu_2

        def dowm_sampling(inputs, pooling_type, use_shortcut, shortcut):
            if pooling_type == 'kmaxpool':
                k = math.ceil(K.int_shape(inputs)[1] / 2)
                pool = KMaxPooling(k)(inputs)
            elif pooling_type == 'maxpool':
                pool = MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)
            elif pooling_type == 'conv':
                pool = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                              kernel_initializer='he_uniform', padding='same')(inputs)
            else:
                raise ValueError('pooling_type `{}` not understood'.format(pooling_type))
            if shortcut is not None and use_shortcut:
                shortcut = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                                  kernel_initializer='he_uniform', padding='same')(shortcut)
                return Add()([pool, shortcut])
            else:
                return pool

        if not isinstance(depth, list) or len(depth) != 4:
            raise ValueError('depth must be a list of 4 integer, got `{}`'.format(depth))
        if pooling_type not in ['conv', 'maxpool', 'kmaxpool']:
            raise ValueError('pooling_type `{}` not understood'.format(pooling_type))

        input_text = Input(shape=(self.max_len, ))

        # Lookup table
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # first temporal conv layer
        conv_out = Conv1D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(text_embed)
        shortcut = conv_out

        # temporal conv block: 64
        for i in range(depth[0]):
            if i < depth[0] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        # shortcut is the second last conv block output
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 128
        for i in range(depth[1]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 256
        for i in range(depth[2]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)

        # temporal conv block: 512
        for i in range(depth[3]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # 8-max pooling
        conv_out = KMaxPooling(k=8)(conv_out)
        flatten = Flatten()(conv_out)

        fc1 = Dense(2048, activation='relu')(flatten)
        fc2 = Dense(2048, activation='relu')(fc1)

        return Model(input_text, fc2)

    # implementation of dynamic convolution network
    def dcnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # wide convolution
        zero_padded_1 = ZeroPadding1D((6, 6))(text_embed)
        conv_1 = Conv1D(filters=128, kernel_size=7, strides=1, padding='valid')(zero_padded_1)
        # dynamic k-max pooling
        k_maxpool_1 = KMaxPooling(int(self.max_len / 3 * 2))(conv_1)
        # non-linear feature function
        non_linear_1 = ReLU()(k_maxpool_1)

        # wide convolution
        zero_padded_2 = ZeroPadding1D((4, 4))(non_linear_1)
        conv_2 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_2)
        # dynamic k-max pooling
        k_maxpool_2 = KMaxPooling(int(self.max_len / 3 * 1))(conv_2)
        # non-linear feature function
        non_linear_2 = ReLU()(k_maxpool_2)

        # wide convolution
        zero_padded_3 = ZeroPadding1D((2, 2))(non_linear_2)
        conv_3 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_3)
        # folding
        folded = Folding()(conv_3)
        # dynamic k-max pooling
        k_maxpool_3 = KMaxPooling(k=10)(folded)
        # non-linear feature function
        non_linear_3 = ReLU()(k_maxpool_3)

        flatten = Flatten()(non_linear_3)

        return Model(input_text, flatten)

    # weighted merge models
    @staticmethod
    def weighted_merge(n_model, n_dim):
        """
        compute a weighted output from multiple parallel models using softmax layer
        Input: concatenate of `n_model` parallel models
        """
        model_concat = Input(shape=(n_model*n_dim, ))

        weights = Dense(n_model, activation='softmax')(model_concat)
        weights_expand = Lambda(lambda x: K.expand_dims(x, axis=-1))(weights)   # [batch_size, n_model, 1]

        model_unfold = Reshape((n_model, n_dim))(model_concat)  # [batch_size, n_model, n_dim]
        weighted_output = multiply([model_unfold, weights_expand])  # [batch_size, n_model, n_dim]
        weighted_output = Lambda(lambda x: K.sum(x, axis=1))(weighted_output)   # [batch_size, n_dim]

        return Model(model_concat, weighted_output)


# callback for sentiment analysis model
class SentiModelMetrics(Callback):
    def __init__(self):
        super(SentiModelMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s_macro = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict([self.validation_data[0], self.validation_data[1]])
        _val_f1_macro = get_score_senti(self.validation_data[1], self.validation_data[2], valid_results)
        logs['val_f1_macro'] = _val_f1_macro
        self.val_f1s_macro.append(_val_f1_macro)
        print('val_f1_macro: %f' % _val_f1_macro)
        return


# model for sentiment analysis
class SentimentModel(object):
    def __init__(self, config, train_data, valid_data, test_data, test_label=None):
        self.config = config
        self.n_class = self.config.n_senti_class
        self.max_len = self.config.max_len[self.config.level]
        self.weights = np.load('data/%s_embeddings.npy' % self.config.level)
        self.lstm_units = 128

        self.x_train, self.aspect_train, self.y_train = train_data
        self.x_train = np.array(self.x_train)
        self.aspect_train = np.array(self.aspect_train)
        self.y_train = np.array(self.y_train)

        self.x_valid, self.aspect_valid, self.y_valid = valid_data
        self.x_valid = np.array(self.x_valid)
        self.aspect_valid = np.array(self.aspect_valid)
        self.y_valid = np.array(self.y_valid)

        self.x_test, self.aspect_test = test_data
        self.x_test = np.array(self.x_test)
        self.aspect_test = np.array(self.aspect_test)

        if test_label is not None:
            self.has_test_label = True
            self.y_test = np.array(test_label)
        else:
            self.has_test_label = False

        self.callbacks = []
        self.init_callbacks()

        self.model = None
        self.build_model(self.config.senti_model_type)

    def init_callbacks(self):
        self.callbacks.append(SentiModelMetrics())

        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_mode,
            verbose=self.config.checkpoint_verbose
        ))

        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience
        ))

    def pad_sequence(self, input_seq):
        return pad_sequences(input_seq, maxlen=self.max_len)

    def load(self):
        print('loading model checkponit {} ...\n'.format('%s.hdf5') % self.config.exp_name)
        self.model.load_weights(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print('Model loaded')

    def build_model(self, senti_model_type):
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1,), name='input_aspect')

        if senti_model_type == 'at':
            base_network = self.at_lstm()
        elif senti_model_type == 'atcu':
            base_network = self.at_lstm_custom()
        elif senti_model_type == 'ae':
            base_network = self.ae_lstm()
        elif senti_model_type == 'atae':
            base_network = self.atae_lstm()
        elif senti_model_type == 'memnet':
            base_network = self.memnet()
        elif senti_model_type == 'ram':
            base_network = self.ram()
        else:
            raise Exception('Senti Model Type `%s` Not Understood' % senti_model_type)

        vector_text = base_network([input_text, input_aspect])
        dense_layer = Dense(256, activation='relu')(vector_text)

        output_layer = Dense(self.n_class, activation='softmax')(dense_layer)

        self.model = Model([input_text, input_aspect], output_layer)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)

    def train(self):
        x_train_pad = self.pad_sequence(self.x_train)
        x_valid_pad = self.pad_sequence(self.x_valid)

        print('start training...')
        self.model.fit(x=[x_train_pad, self.aspect_train], y=self.y_train, batch_size=self.config.batch_size,
                       epochs=self.config.num_epochs, validation_data=([x_valid_pad, self.aspect_valid], self.y_valid),
                       callbacks=self.callbacks)
        print('training end...')

        print('score over valid data:')
        valid_pred = self.model.predict([x_valid_pad, self.aspect_valid])
        get_score_senti(self.aspect_valid, self.y_valid, valid_pred)

    def score(self, input_text, input_aspect, label):
        text_pad = self.pad_sequence(input_text)
        pred = self.model.predict([text_pad, input_aspect])
        get_score_senti(input_aspect, label, pred)

    def predict(self, input_text, input_aspect):
        text_pad = self.pad_sequence(input_text)
        pred = self.model.predict([text_pad, input_aspect])
        pred_classes = np.argmax(pred, axis=-1)
        # change class to polarity
        pred_polaritys = np.array([self.config.senti_class_to_polarity[pred_cls] for pred_cls in pred_classes])
        return pred_polaritys

    # additionally concatenate aspect vector with input sequence vectors, following which a bilstm model's applied.
    def ae_lstm(self):
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1, ), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)    # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        bilstm_layer = Bidirectional(LSTM(self.lstm_units))(input_concat)

        return Model([input_text, input_aspect], bilstm_layer)

    # attention mechanism based bilstm model
    # concatenate aspect vector with hidden vectors output by bilstm model, then attention mechanism is applied to
    # compute attention weights over hidden vectors (in other words, steps)
    def at_lstm(self):
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1,), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        hidden_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(text_embed)    # hidden vectors output by bilstm

        # compute attention weight for each hidden vector (step), refer to https://aclweb.org/anthology/D16-1058
        concat = concatenate([hidden_out, repeat_aspect], axis=-1)
        M = TimeDistributed(Dense(self.config.embedding_dim + self.config.aspect_embed_dim, activation='tanh'))(concat)
        attention = TimeDistributed(Dense(1))(M)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)    # [batch_size, max_len]

        # apply the attention
        repeat_attention = RepeatVector(2*self.lstm_units)(attention)   # [batch_size, 2*units, max_len)
        repeat_attention = Permute((2, 1))(repeat_attention)    # [batch_size, max_len, 2*units]
        sent_representation = multiply([hidden_out, repeat_attention])
        sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

        return Model([input_text, input_aspect], sent_representation)

    # lstm with custom attention layer (considering masking)
    def at_lstm_custom(self):
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1,), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights], mask_zero=True)(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        hidden_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(text_embed)  # hidden vectors output by bilstm
        concat = concatenate([hidden_out, repeat_aspect], axis=-1)  # mask after concatenate will be same as hidden_out's mask

        # apply attention mechanism
        sent_representation = Attention()(concat)

        return Model([input_text, input_aspect], sent_representation)

    # combine ae_lstm and at_lstm
    def atae_lstm(self):
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1,), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        hidden_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(input_concat)

        # compute attention weight for each hidden vector (step), refer to https://aclweb.org/anthology/D16-1058
        concat = concatenate([hidden_out, repeat_aspect], axis=-1)
        M = TimeDistributed(Dense(self.config.embedding_dim + self.config.aspect_embed_dim, activation='tanh'))(concat)
        attention = TimeDistributed(Dense(1))(M)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)  # [batch_size, max_len]

        # apply the attention
        repeat_attention = RepeatVector(2*self.lstm_units)(attention)  # [batch_size, units, max_len)
        repeat_attention = Permute((2, 1))(repeat_attention)  # [batch_size, max_len, units]
        sent_representation = multiply([hidden_out, repeat_attention])
        sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

        return Model([input_text, input_aspect], sent_representation)

    # deep memory network (here we don't use location attention)
    def memnet(self):
        n_hop = 2

        input_text = Input(shape=(self.max_len, ), name='input_text')
        input_aspect = Input(shape=(1, ), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.embedding_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d

        '''
        multi-hop computation layers to learn representation of text with multiple levels of abstraction
        here we only apply content attention without location attention
        '''
        # the parameter of attention and linear layers are shared in different hops
        attention_layer = TimeDistributed(Dense(1, activation='tanh'))
        linear_layer = Dense(self.config.embedding_dim, )

        # output from each computation layer, representing text in different level of abstraction
        computation_layers_out = [aspect_embed]

        for h in range(n_hop):
            # content attention layer
            repeat_out = RepeatVector(self.max_len)(computation_layers_out[-1])     # repeat the last output from last computation layer
            concat = concatenate([text_embed, repeat_out], axis=-1)  # [batch_size, max_len, 2*embed]

            attention = attention_layer(concat)    # [batch_size, max_len, 1]
            attention = Flatten()(attention)
            attention = Activation('softmax')(attention)    # [batch_size, max_len]
            repeat_attention = RepeatVector(self.config.embedding_dim)(attention)  # [batch_size, embed, max_len)
            repeat_attention = Permute((2, 1))(repeat_attention)  # [batch_size, max_len, embed]
            # apply attention
            sent_representation = multiply([text_embed, repeat_attention])
            sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)   # [batch_size, embed]

            # linear layer
            out_linear = linear_layer(computation_layers_out[-1])   # [batch_size, embed]

            computation_layers_out.append(add([sent_representation, out_linear]))

        return Model([input_text, input_aspect], computation_layers_out[-1])

    # recurrent attention network on memory (without location weighted memory)
    def ram(self):
        n_hop = 5

        # input module
        input_text = Input(shape=(self.max_len,), name='input_text')
        input_aspect = Input(shape=(1,), name='input_aspect')

        text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                               weights=[self.weights], mask_zero=True)(input_text)
        text_embed = SpatialDropout1D(0.2)(text_embed)

        aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)    # repeat aspect for every word in sequence

        # memory module
        hidden_out_1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(text_embed)
        hidden_out_2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(hidden_out_1)
        memory = concatenate([hidden_out_2, repeat_aspect], axis=-1)

        # position-weighted memory module: discarded

        # recurrent attention module
        sent_representation = RecurrentAttention(units=self.lstm_units, n_hop=n_hop)(memory)

        return Model([input_text, input_aspect], sent_representation)

    """
    models below are not suitable for our task cause aspect tokens are not in the sentences
    """
    # target dependent lstm
    def td_lstm(self):
        pass

    # target connection lstm
    def tc_lstm(self):
        pass

    # deep memory network with location attention
    def memnet_original(self):
        pass

    # interactive attention network
    def ian(self):
        pass

    # ram memory network with location weighted memory
    def ram_original(self):
        pass

    # implementation of `Content Attention Model for Aspect Based Sentiment Analysis`
    def cabasc(self):
        pass


# callback for joint model
class JointModelMetrics(Callback):
    def __init__(self, n_aspect):
        self.n_aspect = n_aspect
        super(JointModelMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s_macro = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict(self.validation_data[0])
        _val_f1_macro = get_score_joint(self.validation_data[1:self.n_aspect + 1], valid_results)
        logs['val_f1_macro'] = _val_f1_macro
        self.val_f1s_macro.append(_val_f1_macro)
        print('val_f1_macro: %f' % _val_f1_macro)
        return


# joint model for aspect classification and sentiment analysis
class JointModel(object):
    def __init__(self, config, train_data, valid_data, test_text, test_label=None):
        self.config = config
        self.weights = np.load('data/%s_embeddings.npy' % self.config.level)
        self.n_class = self.config.n_senti_class + 1
        self.max_len = self.config.max_len[self.config.level]
        self.lstm_units = 128

        self.x_train, self.y_train = train_data
        self.y_train = np.array(self.y_train)
        self.x_valid, self.y_valid = valid_data
        self.y_valid = np.array(self.y_valid)
        self.x_test = test_text

        if test_label is not None:
            self.has_test_label = True
            self.y_test = np.array(test_label)
        else:
            self.has_test_label = False
            self.y_test = None

        self.callbacks = []
        self.init_callbacks()

        self.model = None
        if isinstance(self.config.joint_model_type, list):
            self.build_merged_model(self.config.joint_model_type, self.config.merge_for_each)
        else:
            self.build_model(self.config.joint_model_type)

    def pad_sequences(self, input_seq):
        if self.config.joint_model_type == 'han':
            return pad_sequences_2d(input_seq, max_sents=self.config.han_max_sent,
                                    max_words=self.config.han_max_sent_len[self.config.level])
        else:
            return pad_sequences(input_seq, maxlen=self.max_len)

    def init_callbacks(self):
        self.callbacks.append(JointModelMetrics(self.config.n_aspect))
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

    def build_base_network(self, joint_model_type):
        if joint_model_type == 'cnn':
            base_network = self.cnn()
        elif joint_model_type == 'multicnn':
            base_network = self.multi_cnn()
        elif joint_model_type == 'bilstm':
            base_network = self.bilstm()
        elif joint_model_type == 'dpcnn':
            base_network = self.dpcnn()
        elif joint_model_type == 'rnncnn':
            base_network = self.rnncnn()
        elif joint_model_type == 'han':
            base_network = self.han()
        elif joint_model_type == 'rcnn':
            base_network = self.rcnn()
        elif joint_model_type == 'rcnn1':
            base_network = self.rcnn_1()
        elif joint_model_type == 'vdcnn':
            base_network = self.vdcnn(self.config.vd_depth, self.config.vd_pooling_type, self.config.vd_use_shortcut)
        elif joint_model_type == 'dcnn':
            base_network = self.dcnn()
        else:
            raise ValueError('Joint Model Type `%s` Not Understood' % joint_model_type)

        return base_network

    def build_model(self, joint_model_type):
        base_network = self.build_base_network(joint_model_type)

        if joint_model_type == 'han':
            input_text = Input(shape=(self.config.han_max_sent, self.config.han_max_sent_len[self.config.level]))
        else:
            input_text = Input(shape=(self.max_len,))
        vector_text = base_network(input_text)
        dense_layer = Dense(256, activation='relu')(vector_text)

        aspect_layers = []
        for aspect_name in self.config.aspect_names:
            # we found adding another dense layer for each aspect not helping
            # dense_layer_sepc = Dense(128, activation='relu')(dense_layer)
            aspect_layers.append(Dense(self.n_class, activation='softmax', name=aspect_name)(dense_layer))

        self.model = Model(input_text, aspect_layers)

        loss = ['categorical_crossentropy'] * self.config.n_aspect
        if self.config.joint_use_loss_weight:
            loss_weights = self.config.joint_loss_weight
        else:
            loss_weights = [1.] * self.config.n_aspect

        self.model.compile(loss=loss, loss_weights=loss_weights, metrics=['accuracy'], optimizer=self.config.optimizer)

    def build_merged_model(self, joint_model_types, merge_for_each):
        if len(joint_model_types) <= 1:
            raise ValueError('Models less than 2: `{}`'.format(joint_model_types))

        models = []
        input_text = Input(shape=(self.max_len,))
        for joint_model_type in joint_model_types:
            if joint_model_type == 'han':
                raise ValueError('han model is not supported')

            base_network = self.build_base_network(joint_model_type)
            vector_text = base_network(input_text)
            dense_layer = Dense(256, activation='relu')(vector_text)

            aspect_layers = []
            for _ in self.config.aspect_names:
                # we found adding another dense layer for each aspect not helping
                # dense_layer_sepc = Dense(128, activation='relu')(dense_layer)
                aspect_layers.append(Dense(self.n_class, activation='softmax')(dense_layer))
            models.append(aspect_layers)

        n_model = len(models)
        if merge_for_each:
            weighted_predict = []
            for aspect_predict in zip(*models):
                aspect_predict = concatenate(list(aspect_predict), axis=-1)   # [batch_size, 4*n_models]
                weighted_predict.append(self.weighted_merge(n_model=n_model, n_dim=self.n_class)(aspect_predict))
        else:
            models = [concatenate(model_predict, axis=-1) for model_predict in models]
            n_aspects = len(self.config.aspect_names)
            n_dim = self.n_class * n_aspects
            weighted_predict = self.weighted_merge(n_model=n_model, n_dim=n_dim)(concatenate(models, axis=-1))

            weighted_predict = Lambda(lambda x: list(tf.split(weighted_predict, n_aspects, axis=-1)))(weighted_predict)

        print(len(weighted_predict))
        self.model = Model(input_text, weighted_predict)

        loss = ['categorical_crossentropy'] * self.config.n_aspect
        if self.config.joint_use_loss_weight:
            loss_weights = self.config.joint_loss_weight
        else:
            loss_weights = [1.] * self.config.n_aspect

        self.model.compile(loss=loss, loss_weights=loss_weights, metrics=['accuracy'], optimizer=self.config.optimizer)

    def train(self):
        x_train_pad = self.pad_sequences(self.x_train)
        x_valid_pad = self.pad_sequences(self.x_valid)
        y_train_list = [self.y_train[:, i*self.n_class:(i+1)*self.n_class] for i in range(self.config.n_aspect)]
        y_valid_list = [self.y_valid[:, i*self.n_class:(i+1)*self.n_class] for i in range(self.config.n_aspect)]

        print('training...')
        self.model.fit(x_train_pad, y_train_list, epochs=self.config.num_epochs, batch_size=self.config.batch_size,
                       validation_data=(x_valid_pad, y_valid_list), callbacks=self.callbacks)
        print('training end...')

        print('score over valid data:')
        valid_pred = self.model.predict(x_valid_pad)
        get_score_joint(y_valid_list, valid_pred)

    def load(self):
        print("Loading model checkpoint {} ...\n".format('%s.hdf5' % self.config.exp_name))
        self.model.load_weights(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        print("Model loaded")

    def score(self, input_data, label):
        input_data_pad = self.pad_sequences(input_data)
        predicts = self.model.predict(input_data_pad)
        label_list = [label[:, i*self.n_class:(i+1)*self.n_class] for i in range(self.config.n_aspect)]
        get_score_joint(label_list, predicts)

    def predict(self, input_data):
        input_data_pad = self.pad_sequences(input_data)
        predicts = self.model.predict(input_data_pad)
        pred_classes = [np.argmax(pred, axis=-1) for pred in predicts]
        # change class to polarity
        pred_polaritys = [list(map(self.config.join_class_to_polarity.get, pred_cls)) for pred_cls in pred_classes]
        return pred_polaritys

    def cnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                strides=1, activation='relu')(embedding_layer)
            maxpooling = MaxPool1D(pool_size=self.max_len - filter_length + 1)(conv_layer)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        concatenate_layer = concatenate(inputs=conv_layers)
        return Model(input_text, concatenate_layer)

    def multi_cnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer_1 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(embedding_layer)
            bn_layer_1 = BatchNormalization()(conv_layer_1)
            conv_layer_2 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(bn_layer_1)
            bn_layer_2 = BatchNormalization()(conv_layer_2)
            maxpooling = MaxPooling1D(pool_size=self.max_len - 2*filter_length + 2)(bn_layer_2)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        concatenate_layer = concatenate(conv_layers)
        return Model(input_text, concatenate_layer)

    def bilstm(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)
        bilstm_layer = Bidirectional(LSTM(128))(embedding_layer)
        dropout = Dropout(self.config.dropout)(bilstm_layer)

        return Model(input_text, dropout)

    def dpcnn(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        repeat = 3
        size = self.max_len
        region_x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(embedding_layer)
        x = Activation(activation='relu')(region_x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Add()([x, region_x])

        for _ in range(repeat):
            px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
            size = int((size + 1) / 2)
            x = Activation(activation='relu')(px)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Activation(activation='relu')(x)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Add()([x, px])

        x = MaxPooling1D(pool_size=size)(x)
        x = Flatten()(x)

        return Model(input_text, x)

    def rnncnn(self):
        input_text = Input(shape=(self.max_len,))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim,
                                    weights=[self.weights])(input_text)
        x = SpatialDropout1D(0.2)(embedding_layer)
        x = Bidirectional(GRU(100, return_sequences=True))(x)

        x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        
        return Model(input_text, conc)

    def han(self):
        def word_encoder():
            input_words = Input(shape=(self.config.han_max_sent_len[self.config.level],))
            word_vectors = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
                                     weights=[self.weights], mask_zero=True)(input_words)
            sent_encoded = Bidirectional(GRU(self.lstm_units, return_sequences=True))(word_vectors)
            return Model(input_words, sent_encoded)

        def sentence_encoder():
            input_sents = Input(shape=(self.config.han_max_sent, self.lstm_units*2))
            sents_masked = Masking()(input_sents)   # support masking
            doc_encoded = Bidirectional(GRU(self.lstm_units, return_sequences=True))(sents_masked)
            return Model(input_sents, doc_encoded)

        input_text = Input(shape=(self.config.han_max_sent, self.config.han_max_sent_len[self.config.level]))
        sent_encoded = TimeDistributed(word_encoder())(input_text)  # word encoder
        sent_vectors = TimeDistributed(Attention(bias=True))(sent_encoded)  # word attention
        doc_encoded = sentence_encoder()(sent_vectors)  # sentence encoder
        doc_vector = Attention(bias=True)(doc_encoded)    # sentence attention

        return Model(input_text, doc_vector)

    # my implementation of recurrent convolution neural network
    # ValueError: An operation has `None` for gradient.
    # Please make sure that all of your ops have a gradient defined (i.e. are differentiable)
    def rcnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights],
                                    mask_zero=True)(input_text)
        x = SpatialDropout1D(0.2)(embedding_layer)
        x = RecurrentCNN(units=self.lstm_units)(x)

        x = TimeDistributed(Dense(self.lstm_units, activation='tanh'))(x)   # equation (4)
        max_pool = GlobalMaxPooling1D()(x)  # equation (5)

        return Model(input_text, max_pool)

    # implementation of recurrent convolution neural network I found from github (slightly modified)
    # see https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier
    def rcnn_1(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)

        doc_embedding = SpatialDropout1D(0.2)(embedding_layer)
        # We shift the document to the right to obtain the left-side contexts
        l_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                     x[:, :-1]], axis=1))(doc_embedding)
        # We shift the document to the left to obtain the right-side contexts
        r_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, 1:]], axis=1))(doc_embedding)

        # use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(self.lstm_units, return_sequences=True)(l_embedding)    # See equation (1)
        backward = LSTM(self.lstm_units, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2)
        # Keras returns the output sequences in reverse order.
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
        together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

        # use conv1D instead of TimeDistributed and Dense
        semantic = Conv1D(self.lstm_units, kernel_size=1, activation="tanh")(together)  # See equation (4).

        pool_rnn = Lambda(lambda x: K.max(x, axis=1))(semantic)     # See equation (5).

        return Model(input_text, pool_rnn)

    # implementation of very deep convolution neural network
    # Reference: https://github.com/zonetrooper32/VDCNN/tree/tensorflow_version (modified)
    def vdcnn(self, depth, pooling_type, use_shortcut):
        def conv_block(inputs, filters, use_shortcut, shortcut):
            conv_1 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(inputs)
            bn_1 = BatchNormalization()(conv_1)
            relu_1 = ReLU()(bn_1)
            conv_2 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(relu_1)
            bn_2 = BatchNormalization()(conv_2)
            relu_2 = ReLU()(bn_2)

            if shortcut is not None and use_shortcut:
                return Add()([inputs, shortcut])
            else:
                return relu_2

        def dowm_sampling(inputs, pooling_type, use_shortcut, shortcut):
            if pooling_type == 'kmaxpool':
                k = math.ceil(K.int_shape(inputs)[1] / 2)
                pool = KMaxPooling(k)(inputs)
            elif pooling_type == 'maxpool':
                pool = MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)
            elif pooling_type == 'conv':
                pool = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                              kernel_initializer='he_uniform', padding='same')(inputs)
            else:
                raise ValueError('pooling_type `{}` not understood'.format(pooling_type))
            if shortcut is not None and use_shortcut:
                shortcut = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                                  kernel_initializer='he_uniform', padding='same')(shortcut)
                return Add()([pool, shortcut])
            else:
                return pool

        if not isinstance(depth, list) or len(depth) != 4:
            raise ValueError('depth must be a list of 4 integer, got `{}`'.format(depth))
        if pooling_type not in ['conv', 'maxpool', 'kmaxpool']:
            raise ValueError('pooling_type `{}` not understood'.format(pooling_type))

        input_text = Input(shape=(self.max_len, ))

        # Lookup table
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # first temporal conv layer
        conv_out = Conv1D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(text_embed)
        shortcut = conv_out

        # temporal conv block: 64
        for i in range(depth[0]):
            if i < depth[0] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        # shortcut is the second last conv block output
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 128
        for i in range(depth[1]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 256
        for i in range(depth[2]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                 shortcut=shortcut)

        # temporal conv block: 512
        for i in range(depth[3]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # 8-max pooling
        conv_out = KMaxPooling(k=8)(conv_out)
        flatten = Flatten()(conv_out)

        fc1 = Dense(2048, activation='relu')(flatten)
        fc2 = Dense(2048, activation='relu')(fc1)

        return Model(input_text, fc2)

    # implementation of dynamic convolution network
    def dcnn(self):
        input_text = Input(shape=(self.max_len, ))
        embedding_layer = Embedding(self.weights.shape[0], self.config.embedding_dim, weights=[self.weights])(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # wide convolution
        zero_padded_1 = ZeroPadding1D((6, 6))(text_embed)
        conv_1 = Conv1D(filters=128, kernel_size=7, strides=1, padding='valid')(zero_padded_1)
        # dynamic k-max pooling
        k_maxpool_1 = KMaxPooling(int(self.max_len / 3 * 2))(conv_1)
        # non-linear feature function
        non_linear_1 = ReLU()(k_maxpool_1)

        # wide convolution
        zero_padded_2 = ZeroPadding1D((4, 4))(non_linear_1)
        conv_2 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_2)
        # dynamic k-max pooling
        k_maxpool_2 = KMaxPooling(int(self.max_len / 3 * 1))(conv_2)
        # non-linear feature function
        non_linear_2 = ReLU()(k_maxpool_2)

        # wide convolution
        zero_padded_3 = ZeroPadding1D((2, 2))(non_linear_2)
        conv_3 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_3)
        # folding
        folded = Folding()(conv_3)
        # dynamic k-max pooling
        k_maxpool_3 = KMaxPooling(k=10)(folded)
        # non-linear feature function
        non_linear_3 = ReLU()(k_maxpool_3)

        flatten = Flatten()(non_linear_3)

        return Model(input_text, flatten)

    # weighted merge models
    @staticmethod
    def weighted_merge(n_model, n_dim):
        """
        compute a weighted output from multiple parallel models using softmax layer
        Input: concatenate of `n_model` parallel models
        """
        model_concat = Input(shape=(n_model*n_dim, ))

        weights = Dense(n_model, activation='softmax')(model_concat)
        weights_expand = Lambda(lambda x: K.expand_dims(x, axis=-1))(weights)   # [batch_size, n_model, 1]

        model_unfold = Reshape((n_model, n_dim))(model_concat)  # [batch_size, n_model, n_dim]
        weighted_output = multiply([model_unfold, weights_expand])  # [batch_size, n_model, n_dim]
        weighted_output = Lambda(lambda x: K.sum(x, axis=1))(weighted_output)   # [batch_size, n_dim]

        return Model(model_concat, weighted_output)

