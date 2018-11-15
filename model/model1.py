# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras import optimizers
from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D


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
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


