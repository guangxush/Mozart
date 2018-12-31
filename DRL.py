# -*- coding: utf-8 -*-
import os
import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from use_all_data import play_game


class DRL:
    def __init__(self):

        # self.env = gym.make('CartPole-v0')
        # self.env = use_all_data.play_game()
        if not os.path.exists('./modfile/rl_model'):
            os.mkdir('./modfile/rl_model')

        if not os.path.exists('./data/history'):
            os.mkdir('./data/history')

    def play(self, m='dpg'):
        """play game with model.
        """
        print('play...')
        # 初始化环境
        # observation = self.env.reset()
        # i = 0
        observation, _, _, _ = play_game(0)
        # i += 1

        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            # 渲染图像
            # self.env.render()

            x = observation.reshape(-1, 1)
            if m == 'dpg':
                prob = self.model.predict(x)[0][0]
                action = 1 if prob > 0.5 else 0
            else:
                action = np.argmax(self.model.predict(x)[0])
            # 执行随机的action 获得返回值
            # observation, reward, done, _ = self.env.step(action)
            observation, reward, done, _ = play_game(random.randint(1, 90)+action)
            # i += 1
            # 计算回报值
            # print(done)
            reward_sum += reward
            done = True if done == 1 else False
            if done:
                print("Reward for this episode was: {}".format(reward_sum))
                random_episodes += 1
                reward_sum = 0
                # 重启环境
                # observation = self.env.reset()
        # 关闭环境
        # self.env.close()

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
