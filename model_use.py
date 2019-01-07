# -*- encoding:utf-8 -*-

from model.model2 import mlp2
from util.data_load import load_data3
import numpy as np
from util.util import cal_err_ratio
from util.data_load import generate_imdb_model2_data, generate_imdb_model2_data_rl


# use this model test model1&model2 or generate the result
def model_use(i):
    filepath = "./modfile/model2file/imdb.mlp.best_model.h5"
    result_path = './data/model2_result/imdb_' + str(i) + '_data.csv'
    model_file = './modfile/model1file/lstm.best_model_'
    # test_pos_file = './data/part_data/test_pos_1.txt'
    # test_neg_file = './data/part_data/test_neg_1.txt'
    test_file = './data/part_data_all/test_1.txt'
    generate_imdb_model2_data(model_file=model_file, result_path=result_path,
                              test_file=test_file, count=10)
    print('Load result ...')
    x_test, y_test = load_data3(data_path=result_path)
    model2 = mlp2(sample_dim=x_test.shape[1], class_count=2)
    model2.load_weights(filepath)
    results = model2.predict(x_test)
    label = np.argmax(results, axis=1).astype('int')
    cal_err_ratio(file_name='test', label=label, y_test=y_test)


def model_use_rl(i):
    filepath = "./modfile/model2file/imdb.mlp.best_model.h5"
    result_path = './data/model2_result/imdb_' + str(i) + '_data.csv'
    model_file = './modfile/model1file/lstm.best_model_'
    # test_pos_file = './data/part_data/test_pos_1.txt'
    # test_neg_file = './data/part_data/test_neg_1.txt'
    test_file = './data/part_data_all/test_1.txt'
    generate_imdb_model2_data_rl(model_file=model_file, result_path=result_path,
                                 test_file=test_file, count=10)
    print('Load result ...')
    x_test, y_test = load_data3(data_path=result_path)
    model2 = mlp2(sample_dim=x_test.shape[1], class_count=2)
    model2.load_weights(filepath)
    results = model2.predict(x_test)
    label = np.argmax(results, axis=1).astype('int')
    cal_err_ratio(file_name='test', label=label, y_test=y_test)


if __name__ == '__main__':
    for i in range(1, 10):
        model_use(i=i)