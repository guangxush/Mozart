# -*- encoding:utf-8 -*-
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from model.model1 import mlp1
from data_load import load_testset


def load_data(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'iris.data'), header=None)
    train_dataframe.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_label']
    print(train_dataframe['class_label'])
    class_le = LabelEncoder()
    train_dataframe['class_label'] = class_le.fit_transform(train_dataframe['class_label'].values)
    train_dataframe_new = train_dataframe.sample(frac=1).reset_index(drop=True)
    for i in range(5):
        line = i * 30
        split_dataframe = train_dataframe_new.iloc[line:line+30]
        split_dataframe.to_csv("../data/iris_"+str(i+1)+"_data.csv", encoding='utf-8', header=1, index=0)


def generate_model2_label(file_name, x_test):
    mlp_model = mlp1(sample_dim=x_test.shape[1], class_count=3)
    mlp_model.load_weights(file_name)
    results = mlp_model.predict(x_test)
    print(results)
    label = np.argmax(results, axis=1)
    # print(label)
    return label
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


def generate_model2_data(result_path):
    x_test, y_test = load_testset(data_path='../data/iris_3_data.csv')
    print(x_test)
    y1_test = np.array(generate_model2_label(file_name='../modfile/mlp.best_model.h5', x_test=x_test).tolist())
    print(y1_test)
    print(type(y1_test))
    y2_test = np.array(generate_model2_label(file_name='../modfile/mlp1.best_model.h5', x_test=x_test).tolist())
    print(y2_test)
    y3_test = np.array(generate_model2_label(file_name='../modfile/mlp2.best_model.h5', x_test=x_test).tolist())
    print(y3_test)
    y4_test = np.array(generate_model2_label(file_name='../modfile/mlp3.best_model.h5', x_test=x_test).tolist())
    print(y4_test)
    y5_test = np.array(generate_model2_label(file_name='../modfile/mlp4.best_model.h5', x_test=x_test).tolist())
    print(y5_test)
    z_data = np.c_[y1_test, y2_test, y3_test, y4_test, y5_test, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'test']
    print(z_dataset)
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


if __name__ == '__main__':
    # load_data(data_path='../raw_data/')
    generate_model2_data(result_path='../model2_data/iris_3_data.csv')