# -*- encoding:utf-8 -*-
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from model.model1 import mlp1
from data_load import load_testset


# split the raw data to 5 files in model1 or test floders
def split_raw_data(data_path, out_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'horseColicTraining.txt'), header=None, sep='\t')
    train_dataframe.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T', 'U', 'class_label']
    print(train_dataframe.shape[0])
    print(train_dataframe.shape[1])
    class_le = LabelEncoder()
    train_dataframe['class_label'] = class_le.fit_transform(train_dataframe['class_label'].values)
    train_dataframe_new = train_dataframe.sample(frac=1).reset_index(drop=True)
    total_num = 300
    partition = 5
    group_count = int(total_num/partition)
    for i in range(partition):
        line = i * group_count
        split_dataframe = train_dataframe_new.iloc[line:line+group_count]
        split_dataframe.to_csv("../data/"+out_path+"/horse_"+str(i+1)+"_data.csv", encoding='utf-8', header=1, index=0)


# save the error dataset depends on the model result and the truth label
def make_err_dataset(result_path, label, x_test, y_test):
    count = 0
    err_data_list = []
    for i in label:
        if i != y_test[count]:
            err_result = np.append(x_test[count], y_test[count]).tolist()
            err_data_list.append(err_result)
        count += 1
    err_data = pd.DataFrame(err_data_list)
    err_data.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'class_label']
    err_data.to_csv(result_path, encoding='utf-8', header=1, index=0)


# generate the model labels from model1 result
def generate_model2_label(file_name, mlp_model, x_test):
    group_count = 60
    if not os.path.exists(file_name):
        print(file_name)
        print("file not found!")
        # if file not exists, return [0]*30
        return np.array([0] * group_count)
    mlp_model.load_weights(file_name)
    results = mlp_model.predict(x_test)
    label = np.argmax(results, axis=1)
    # print(label)
    return label
    # make_model2_dataset(result_path='./err_data/iris_1_error_data.csv', label=label, x_test=x_test, y_test=y_test)


# generate model2 dataset by load model1 to predict temp result
def generate_model2_data(data_path, result_path):
    x_test, y_test = load_testset(data_path=data_path)
    mlp_model = mlp1(sample_dim=x_test.shape[1])
    y1_test = np.array(generate_model2_label(file_name='./modfile/model1file/mlp1.best_model.h5', mlp_model=mlp_model,
                                             x_test=x_test).tolist())
    y2_test = np.array(generate_model2_label(file_name='./modfile/model1file/mlp2.best_model.h5', mlp_model=mlp_model,
                                             x_test=x_test).tolist())
    y3_test = np.array(generate_model2_label(file_name='./modfile/model1file/mlp3.best_model.h5', mlp_model=mlp_model,
                                             x_test=x_test).tolist())
    y4_test = np.array(generate_model2_label(file_name='./modfile/model1file/mlp4.best_model.h5', mlp_model=mlp_model,
                                             x_test=x_test).tolist())
    y5_test = np.array(generate_model2_label(file_name='./modfile/model1file/mlp5.best_model.h5', mlp_model=mlp_model,
                                             x_test=x_test).tolist())
    z_data = np.c_[y1_test, y2_test, y3_test, y4_test, y5_test, y_test]
    z_dataset = pd.DataFrame(z_data)
    z_dataset.columns = ['test1', 'test2', 'test3', 'test4', 'test5', 'test']
    z_dataset.to_csv(result_path, encoding='utf-8', header=1, index=0)
    return z_dataset


if __name__ == '__main__':
    split_raw_data(data_path='../raw_data/', out_path="model1_data")
    split_raw_data(data_path='../raw_data/', out_path="test_data")
    # generate_model2_data(data_path='../data/model1_data/iris_3_data.csv',
    #                      result_path='../data/model2_data/iris_3_data.csv')