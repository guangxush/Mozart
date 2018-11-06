# -*- encoding:utf-8 -*-

from util.data_process import generate_model2_data
from model.model2 import mlp2
from util.data_load import load_data3
import numpy as np
from util.util import cal_err_ratio
from util.data_process import generate_model2_data


def model_use(i):
    data_path = './data/test_data/iris_'+str(i)+'_data.csv'
    filepath = "./modfile/model2file/mlp.best_model.h5"
    result_path = './data/test_model2_data/iris_'+str(i)+'_data.csv'
    generate_model2_data(data_path=data_path, result_path=result_path)
    x_train, y_train = load_data3(data_path=result_path)
    mlp_model = mlp2(sample_dim=x_train.shape[1], class_count=3)
    mlp_model.load_weights(filepath)
    results = mlp_model.predict(x_train)
    print(results)
    label = np.argmax(results, axis=1)
    print("pred:")
    print(label)
    print("true:")
    print(np.argmax(y_train, axis=1))
    cal_err_ratio(file_name='test', label=label, y_test=np.argmax(y_train, axis=1))


if __name__ == '__main__':
    for i in range(1, 6):
        model_use(i=i)