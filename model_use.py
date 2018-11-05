# -*- encoding:utf-8 -*-
from util.data_process import generate_model2_data
from model.model2 import mlp2
from util.data_load import load_data2
import numpy as np


def model_use():
    data_path = './model2_data/iris_2_data.csv'
    filepath = "./model2file/mlp.best_model.h5"
    x_train, y_train, x_test, y_test = load_data2(data_path=data_path)
    mlp_model = mlp2(sample_dim=x_test.shape[1], class_count=3)
    mlp_model.load_weights(filepath)
    results = mlp_model.predict(x_train)
    print(results)
    label = np.argmax(results, axis=1)
    print(label)
    return label


if __name__ == '__main__':
    model_use()