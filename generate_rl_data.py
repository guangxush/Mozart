# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np


def rl_data(data_path, j):
    dataframe = pd.read_csv(data_path, header=0)
    observation = []
    reward = []
    done = []
    _ = []
    _.append(np.array([0, 0, 0, 0]))
    for i in range(10):
        result_data = np.array(dataframe.ix[j, 5 * i:5 * (i + 1)])
        flag = np.array(dataframe.ix[j, -1])
        # print(result_data)
        # print(flag)
        if result_data[4] == flag:
            reward.append(1)
        else:
            reward.append(0)
        if i >= 9:
            done.append(1)
            observation.append(_[i])
        else:
            done.append(0)
            observation.append(_[i])
            _.append(result_data[0:4])

    return observation, reward, done, _


if __name__ == '__main__':
    data_path = './data/model2_result/imdb_rl_9_data.csv'
    observation, reward, done, _ = rl_data(data_path, 0)
    print(observation)
    print(reward)
    print(done)
    print(_)
