# -*- encoding:utf-8 -*-

from keras import Sequential
from keras.layers import Dense, Dropout
from keras import losses


def mlp(sample_dim):
    model = Sequential()
    model.add(Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss=losses.mae, optimizer='adam')
    return model


if __name__ == '__main__':
    print()
