# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from model.model1 import discriminate_model


def train():
    data_path = './data/model2_result/imdb_rl_9_data.csv'
    dataframe = pd.read_csv(data_path, header=0)
    X_train1 = dataframe.ix[:, 0:4]
    Y_label1 = np.array([0 for i in range(len(X_train1))])
    X_train2 = dataframe.ix[:, 4:8]
    Y_label2 = np.array([1 for i in range(len(X_train1))])
    X_train = np.vstack((X_train1, X_train2))
    Y_label = np.concatenate((Y_label1, Y_label2), axis=0)
    print(X_train)
    print(Y_label)
    monitor = 'val_acc'
    filepath = "./modfile/discriminate/mlp.best_model.h5"
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=5)
    csv_logger = CSVLogger('logs/imdb_model_discriminate.log')
    model = discriminate_model()
    model.fit(X_train, Y_label, batch_size=128, epochs=100, verbose=1, shuffle=True,
              validation_split=0.3, callbacks=[check_pointer, early_stopping, csv_logger])
    return


def discriminate():
    return


if __name__ == '__main__':
    train()