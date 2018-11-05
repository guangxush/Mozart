# -*- encoding:utf-8 -*-
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import datetime, time


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


if __name__ == '__main__':
    load_data(data_path='../raw_data/')