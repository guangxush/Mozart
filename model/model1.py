# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras import optimizers
from keras.layers import Dense, Input, Dropout
from keras.models import Model


def mlp1(sample_dim, class_count=7):
    feature_input = Input(shape=(sample_dim,), name='mlp_input')

    x = Dense(256, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(feature_input)
    x = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    output = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=[feature_input],
                  outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
    return model

