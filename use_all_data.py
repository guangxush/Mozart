from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util import data_process
from util.util import cal_err_ratio_only
from model.model1 import lstm_mul_model_all
import numpy as np
import linecache
import random
from keras.models import Model
import pickle as pkl


def model1():
    results_flag = True
    train_file = "./data/part_data_all/all_train.txt"
    test_file = "./data/part_data_all/all_test.txt"
    # train model1
    monitor = 'val_acc'
    filepath = "./modfile/model1file/all1.lstm.best_model.h5"
    check_pointer = ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=1,
                                    save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=3)
    csv_logger = CSVLogger('logs/all_imdb_model2_mlp.log')
    Xtrain, Xtest, ytrain, ytest = data_process.get_imdb_part_data(raw_file=train_file)
    vocab_size = data_process.get_imdb_vocab_size(train_file)
    model = lstm_mul_model_all(vocab_size=vocab_size)
    model.fit(Xtrain, ytrain, batch_size=32, epochs=50, validation_data=(Xtest, ytest), verbose=1, shuffle=True,
              callbacks=[check_pointer, early_stopping, csv_logger])
    dense4_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_4').output)
    if results_flag:
        print('test the model ...')
        X, y = data_process.get_imdb_test_data(test_file)
        vocab_size = data_process.get_imdb_vocab_size(train_file)
        lstm_model = lstm_mul_model_all(vocab_size)
        lstm_model.load_weights(filepath)
        # 以这个model的预测值作为输出
        dense4_output = dense4_layer_model.predict(X)
        print(dense4_output)
        results = lstm_model.predict_classes(X)
        cal_err_ratio_only(label=results, y_test=y)
    print('***** End Model1 Train *****')


def all_model_use():
    test_file = "./data/part_data_all/all_test.txt"
    filepath = "./modfile/model1file/all1.lstm.best_model.h5"
    train_file = "./data/part_data_all/all_train.txt"
    X, y = data_process.get_imdb_test_data(test_file)
    vocab_size = data_process.get_imdb_vocab_size(train_file)
    lstm_model = lstm_mul_model_all(vocab_size)
    lstm_model.load_weights(filepath)
    dense4_layer_model = Model(inputs=lstm_model.input,
                               outputs=lstm_model.get_layer('Dense_4').output)
    results = lstm_model.predict_classes(X)
    dense4_output = dense4_layer_model.predict(X)
    # status = lstm_model.predict(X) 只有1维
    rl_data = "./data/rl_data.pkl"
    fw = open(rl_data, 'wb')
    result_data_pkl = []
    for i in range(len(results)-1):
        reward = 1 if y[i] == results[i] else 0
        result_data_pkl.append([dense4_output[i], results[i][0], reward, dense4_output[i+1]])
    pkl.dump(result_data_pkl, fw)
    fw.close()


def game(line_number):
    rl_data = "./data/rl_data.pkl"
    return get_line_context(rl_data, line_number)


def get_line_context(rl_data, line_number):
    return linecache.getline(rl_data, line_number).strip()


# 读取某一行数据
def play_game(i):
    if i > 98:
        i = random.randint(1, 50)
    rl_data = "./data/rl_data.pkl"
    pkl_file = open(rl_data, 'rb')
    result_data_pkl = pkl.load(pkl_file)
    observation, reward, done, _ = result_data_pkl[i]
    observation = np.array(observation)
    pkl_file.close()
    return observation, reward, int(done), _


def test_get_line_data():
    observation, reward, done, _ = play_game(3)
    print(np.array(observation))
    print(reward)
    print(done)
    print(np.array(_))


if __name__ == '__main__':
    # model1()
    # all_model_use()
    # print(game(2))
    # print(read_csv(10))
    test_get_line_data()
