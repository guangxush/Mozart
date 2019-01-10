# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_rl_data import rl_data


class DRL:
    def __init__(self):

        # 初始化文件路径
        if not os.path.exists('./modfile/rl_model'):
            os.mkdir('./modfile/rl_model')

        if not os.path.exists('./data/history'):
            os.mkdir('./data/history')

    def play(self, m='dpg'):
        """play game with model.
        """
        print('play...')
        # 初始化环境
        data_path = './data/model2_result/imdb_rl_9_data.csv'
        i = random.randint(1, 90)
        print(i)
        Observation, Reward, Done, _O = rl_data(data_path, i)
        observation, _, _, _ = Observation[0], 0, 0, 0
        # 回报累积值
        reward_sum = 0
        # 游戏次数
        random_episodes = 0
        j = 0
        while random_episodes < 10:
            # 渲染图像
            # 输入向量坐标
            x = observation.reshape(-1, 4)
            if m == 'dpg':
                # 预测概率
                prob = self.model.predict(x)[0][0]
                # print(prob)
                # 动作
                action = 2 if prob > 0.5 else 1
            else:
                # 选区一个概率最大的动作
                action = np.argmax(self.model.predict(x)[0]) + 1
            # 执行随机的action 获得返回值
            # j = random.randint(1, 9)+action
            # print(action)
            j = j + action
            if j >= 10:
                j -= 1
            print(j)
            observation, reward, done, _ = Observation[j], Reward[j], Done[j], _O[j]
            # print(reward)
            # 计算回报值
            reward_sum += reward
            done = True if done == 1 else False
            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                random_episodes += 1
                reward_sum = 0
                j = 0
                # 重启环境
        # 关闭环境

    def try_gym(self, m='dpg'):
        print('use...')
        data_path = './data/model2_result/imdb_rl_9_data.csv'
        i = random.randint(1, 90)
        Observation, Reward, Done, _O = rl_data(data_path, i)
        observation, _, _, _ = Observation[0], 0, 0, 0

        # episodes of game
        random_episodes = 0
        # sum of reward of game per episode
        reward_sum = 0
        j = 0
        while random_episodes < 10:
            # show game
            # random choice a action
            # execute the action
            x = observation.reshape(-1, 4)
            if m == 'dpg':
                # 预测概率
                prob = self.model.predict(x)[0][0]
                print(prob)
                # 动作
                action = 2 if prob > 0.5 else 1
                # print(action)
            else:
                # 选区一个概率最大的动作
                action = 1 + np.argmax(self.model.predict(x)[0])
            # observation, reward, done, _ = play_game(random.randint(1, 90)+action)
            # j = random.randint(1, 9) + action
            j = j + action
            if j >= 10:
                j -= 1
            observation, reward, done, _ = Observation[j], Reward[j], Done[j], _O[j]
            reward_sum += reward
            # print result and reset game env if game done.
            if done:
                random_episodes += 1
                print("Reward for this episode was: {}".format(reward_sum))
                reward_sum = 0
                j = 0

    def plot(self, history):
        x = history['episode']
        r = history['Episode_reward']
        l = history['Loss']

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(x, r)
        ax.set_title('Episode_reward')
        ax.set_xlabel('episode')
        ax = fig.add_subplot(122)
        ax.plot(x, l)
        ax.set_title('Loss')
        ax.set_xlabel('episode')

        plt.show()

    def save_history(self, history, name):
        name = os.path.join('./data/history', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
