# coding:utf-8
from keras.layers.core import Dropout, Activation, Permute
from keras.layers.merge import concatenate, multiply
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, \
    RepeatVector
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Lambda
from keras import backend as K


def creat_Model_BiLSTM_Attention(sourcevocabsize, targetvocabsize, word_W,
                                 input_seq_lenth, output_seq_lenth, emd_dim,
                                 sourcecharsize, char_W, input_word_length, char_emd_dim, batch_size):
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')
    char_embedding = Embedding(input_dim=sourcecharsize, output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)
    char_cnn = TimeDistributed(Conv1D(100, 2, activation='relu', padding='valid'))(char_embedding2)
    # char_lstm = TimeDistributed(LSTM(50, return_sequences=False))(char_embedding2)
    # char_macpool = Dropout(0.5)(char_lstm)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W])(word_input)
    word_embedding = Dropout(0.5)(word_embedding)
    embedding = concatenate([word_embedding, char_macpool], axis=-1)

    BiLSTM0 = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM0 = Dropout(0.5)(BiLSTM0)
    BiLSTM = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(BiLSTM0)
    # BiLSTM = BatchNormalization()(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)

    attention = Dense(1, activation='tanh')(BiLSTM)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(200)(attention)
    attention = Permute([2, 1])(attention)
    # apply the attention
    representation = multiply([BiLSTM, attention])
    representation = BatchNormalization(axis=1)(representation)
    representation = Dropout(0.5)(representation)
    representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)

    output = Dense(targetvocabsize, activation='softmax')(representation)
    Models = Model([word_input, char_input], output)
    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
    return Models


if __name__ == "__main__":
    batch_size = 128