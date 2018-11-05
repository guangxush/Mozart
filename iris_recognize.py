# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_data, make_err_dataset
from model.model1 import mlp
import numpy as np


if __name__ == '__main__':
    results_flag = True

    print('***** Start Train *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data(data_path='data')

    print('Training MLP model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath="./modfile/mlp.best_model.h5", monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    csv_logger = CSVLogger('logs/mlp.log')
    mlp_model = mlp(sample_dim=x_train.shape[1], class_count=3)
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    '''
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('Generate submission ...')
        mlp_model.load_weights(filepath='./modfile/mlp.best_model.h5')
        results = mlp_model.predict(x_test)
        label = np.argmax(results, axis=1)
        print(label)
        print(y_test)
        make_err_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)

    print('***** End Train *****')
