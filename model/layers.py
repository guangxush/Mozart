# -*- coding: utf-8 -*-

import math
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.contrib.framework import sort


# modified based on https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
 e: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, bias=False, return_score=False, **kwargs):
        self.supports_masking = True
        self.return_score = return_score

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = Attention.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_score:
            return K.sum(weighted_input, axis=1), a
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)


class RecurrentAttention(Layer):
    """
    Multiple attentions non-linearly combined with a recurrent neural network (gru) .
    Supports Masking.
    Follows the work of Peng et al. [http://aclweb.org/anthology/D17-1047]
    "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"
    """

    def __init__(self, units, n_hop=5, return_scores=False, initializer='orthogonal', regularizer=None, constraint=None, **kwargs):
        self.units = units
        self.n_hop = n_hop
        self.return_scores = return_scores

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(RecurrentAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception('Input to RecurrentAttention must be a 3D tensor.')

        # gru weights
        self.gru_wr = self.add_weight((input_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wr'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_ur = self.add_weight((self.units, self.units), initializer=self.initializer,
                                      name='{}_ur'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wz = self.add_weight((input_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wz'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_uz = self.add_weight((self.units, self.units), initializer=self.initializer,
                                      name='{}_uz'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wx = self.add_weight((input_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wx'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wg = self.add_weight((self.units, self.units), initializer=self.initializer,
                                      name='{}_wg'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)

        # attention weights
        self.al_w = self.add_weight((self.n_hop, input_shape[-1]+self.units, 1), initializer=self.initializer,
                                    name='{}_al_w'.format(self.name), regularizer=self.regularizer,
                                    constraint=self.constraint)
        self.al_b = self.add_weight((self.n_hop, 1), initializer='zero', name='{}_al_b'.format(self.name),
                                    regularizer=self.regularizer, constraint=self.constraint)

        super(RecurrentAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        scores = []
        batch_size = K.shape(inputs)[0]
        time_steps = K.shape(inputs)[1]
        e = K.zeros(shape=(batch_size, self.units))
        for h in range(self.n_hop):
            # compute attention weight
            repeat_e = K.repeat(e, time_steps)    # [batch_size, time_steps, units]
            inputs_concat = K.concatenate([inputs, repeat_e], axis=-1)    # [batch_size, time_steps, hidden+units]
            g = K.squeeze(K.dot(inputs_concat, self.al_w[h]), axis=-1) + self.al_b[h]   # [batch_size, time_steps]
            a = K.exp(g)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                a *= K.cast(mask, K.floatx())

            a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
            scores.append(a)

            # apply attention
            a = K.expand_dims(a)    # [batch_size, time_steps, 1]
            i_AL = K.sum(inputs * a, axis=1)   # [batch_size, hidden], i_AL is the input of gru at time `h`

            # gru implementation
            r = K.sigmoid(K.dot(i_AL, self.gru_wr) + K.dot(e, self.gru_ur))    # reset gate
            z = K.sigmoid(K.dot(i_AL, self.gru_wz) + K.dot(e, self.gru_uz))    # update gate
            _e = K.tanh(K.dot(i_AL, self.gru_wx) + K.dot(r*e, self.gru_wg))
            e = (1 - z) * e + z * _e  # update e

        if self.return_scores:
            return e, K.concatenate(scores, axis=0)
        else:
            return e

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.return_scores:
            return [(input_shape[0], self.units), (self.n_hop, input_shape[0], input_shape[1])]
        else:
            return input_shape[0], self.units


class KMaxPooling(Layer):
    """
    Implemetation of temporal k-max pooling layer, which was first proposed in Kalchbrenner et al.
    [http://www.aclweb.org/anthology/P14-1062]
    "A Convolutional Neural Network for Modelling Sentences"
    This layer allows to detect the k most important features in a sentence, independent of their
    specific position, preserving their relative order.
    """
    def __init__(self, k=1, **kwargs):
        self.k = k

        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into KMaxPooling muse be a 3D tensor!')
        if self.k > input_shape[1]:
            raise ValueError('detect `%d` most important features from `%d` timesteps is not allowed' %
                             (self.k, input_shape[1]))
        super(KMaxPooling, self).build(input_shape)

    def call(self, inputs):
        """
        Reference: https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        The key point is preserving the relative order
        """
        permute_inputs = K.permute_dimensions(inputs, (0, 2, 1))
        flat_permute_inputs = tf.reshape(permute_inputs, (-1,))
        topk_indices = sort(tf.nn.top_k(permute_inputs, k=self.k)[1])

        all_indices = tf.reshape(tf.range(K.shape(flat_permute_inputs)[0]), K.shape(permute_inputs))
        to_sum_indices = tf.expand_dims(tf.gather(all_indices, 0, axis=-1), axis=-1)
        topk_indices += to_sum_indices

        flat_topk_indices = tf.reshape(topk_indices, (-1, ))
        topk_output = tf.reshape(tf.gather(flat_permute_inputs, flat_topk_indices), K.shape(topk_indices))

        return K.permute_dimensions(topk_output, (0, 2, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k, input_shape[-1]


class Folding(Layer):
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into Folding Layer must be a 3D tensor!')
        super(Folding, self).build(input_shape)

    def call(self, inputs):
        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors.
        # will raise ValueError if K.int_shape(inputs) is odd
        splits = tf.split(inputs, int(K.int_shape(inputs)[-1] / 2), axis=-1)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=-1) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=-1)
        return row_reduced

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / 2)