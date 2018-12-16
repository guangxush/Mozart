from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util import data_process
from util.util import cal_err_ratio_only
from model.model1 import lstm_mul_model


def model1():
    results_flag = True
    train_file = "./data/part_data_all/all_train.txt"
    test_file = "./data/part_data_all/all_test.txt"
    # train model1
    monitor = 'val_acc'
    filepath = "./modfile/model1file/all.lstm.best_model.h5"
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=3)
    csv_logger = CSVLogger('logs/all_imdb_model2_mlp.log')
    Xtrain, Xtest, ytrain, ytest = data_process.get_imdb_part_data(raw_file=train_file)
    vocab_size = data_process.get_imdb_vocab_size(train_file)
    model = lstm_mul_model(vocab_size=vocab_size)
    # model.fit(Xtrain, ytrain, batch_size=32, epochs=50, validation_data=(Xtest, ytest), verbose=1, shuffle=True,
    #           callbacks=[check_pointer, early_stopping, csv_logger])
    if results_flag:
        print('test the model ...')
        X, y = data_process.get_imdb_test_data(test_file)
        vocab_size = data_process.get_imdb_vocab_size(train_file)
        lstm_model = lstm_mul_model(vocab_size)
        lstm_model.load_weights(filepath)
        results = lstm_model.predict_classes(X)
        cal_err_ratio_only(label=results, y_test=y)
    print('***** End Model1 Train *****')


if __name__ == '__main__':
    model1()
