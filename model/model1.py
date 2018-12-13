# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras import optimizers
from keras.layers import Dense, Input, Embedding, TimeDistributed, GlobalMaxPooling1D, Bidirectional\
     , RepeatVector, Conv1D, LSTM
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda
from keras.layers.core import Dropout, Activation, Permute
from keras.layers.merge import concatenate, multiply
from keras import backend as K


def mlp1(sample_dim, class_count=3):
    feature_input = Input(shape=(sample_dim,), name='mlp_input')

    x = Dense(256, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(feature_input)
    x = Dense(128, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(x)
    x = Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(32, kernel_initializer='glorot_uniform', activation='relu')(x)
    output = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=[feature_input],
                  outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
    return model


def cnn(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
    return model


def BiLSTM_Attention(sourcevocabsize, targetvocabsize, word_W,input_seq_lenth, output_seq_lenth, emd_dim,
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
    # embedding = word_embedding

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
    Models.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
    # K.clear_session()
    return Models


def lstm_model():
    model = Sequential()
    model.add(Embedding(89483, 256, input_length=800))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    batch_size = 128


