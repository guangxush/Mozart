# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_all_data
from model.model1 import mlp1
from util.util import cal_err_ratio
import numpy as np


def model():
    results_flag = True
    train_file = './data/total_data/wine_total_data.csv'
    model_file = './modfile/totalfile/mlp.best_model.h5'
    print('***** Start Model1 Train *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_all_data(train_file=train_file)

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=model_file, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=20)
    csv_logger = CSVLogger('logs/model_total_mlp.log')
    mlp1_model = mlp1(sample_dim=x_train.shape[1], class_count=3)
    mlp1_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
                   callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        mlp1_model.load_weights(filepath=model_file)
        results = mlp1_model.predict(x_test)
        label = np.argmax(results, axis=1)
        y_label = y_test
        print("pred:", end='')
        print(label)
        print("true:", end='')
        print(y_label)
        cal_err_ratio(file_name='total_train', label=label, y_test=y_label)
    print('***** End Model Train *****')


if __name__ == '__main__':
    model()