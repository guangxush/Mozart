# -*- encoding:utf-8 -*-

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util.data_load import load_all_data
from model.model1 import cnn
from util.util import cal_err_ratio
import numpy as np


def model():
    results_flag = True
    model_file = './modfile/totalfile/cnn.best_model.h5'
    print('***** Start Model1 Train *****')
    print('Loading data ...')
    X_train, y_train, X_test, y_test, num_classes = load_all_data()

    print('Training CNN model ...')
    monitor = 'val_acc'
    check_pointer = ModelCheckpoint(filepath=model_file, monitor=monitor, verbose=0,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=20)
    csv_logger = CSVLogger('logs/model_total_cnn.log')
    cnn_model = cnn(num_classes=num_classes)
    cnn_model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1, shuffle=True, validation_split=0.2,
                  callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        cnn_model.load_weights(filepath=model_file)
        results = cnn_model.predict(X_test)
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