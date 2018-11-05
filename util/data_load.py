# -*- encoding:utf-8 -*-
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_data(data_path):
    train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'), header=0, index=0)


if __name__ == '__main__':
    load_data(data_path='../data/')