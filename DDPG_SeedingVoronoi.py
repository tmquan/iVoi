from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import random
import shutil
import logging
import argparse
from natsort import natsorted

import cv2
import skimage.io
import skimage.measure
import numpy as np 

# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

# # Using tensorflow
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

# An efficient dataflow loading for training and testing
import tensorpack.dataflow as df
from tensorpack import *
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
# parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=2020, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--max_length_of_trajectory', default=2000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=10, type=int)
args = parser.parse_args()

SHAPE = 256


# Choose the GPU
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.seed:
# Init the randomness
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

import abc
import gym

class CustomEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    # reward_range = (0.0, 1.0)
    @abc.abstractmethod
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Init CustomEnvironment")
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment state and returns an initial observation
        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation

    @abc.abstractmethod
    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        pass

class VoronoiEnvironment(CustomEnvironment):
    def __init__(self, datadir='data', size=100, istrain=True):
        self.size = size

        self._load_images(datadir=datadir)
        self.reset()

    def _load_images(self, datadir=None):
        self.datadir = datadir
        
        # Query the image
        self.imagedir = os.path.join(datadir, 'train')
        self.imagefiles = natsorted(glob.glob(self.imagedir + '/*.*'))
        
        self.size = min(self.size, len(self.imagefiles))

        self.imagefiles = self.imagefiles[:self.size]
        self.images = []
        for imagefile in self.imagefiles:
            print(imagefile)
            image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            # image = cv2.resize(image, (SHAPE, SHAPE))
            self.images.append(image)
        self.images = np.array(np.squeeze(self.images))
        
    def _get_n_masks(self, membr):
        self.coords = []
        label = skimage.measure.label(membr)
        areas = [r.area for r in skimage.measure.regionprops(label)]
        areas.sort()
        if len(areas) > 2:
            for region in skimage.measure.regionprops(label):
                if True:# if region.area < 6 and region.area > 3:
                    # print(region.centroid)
                    # print(region.area)
                    centroid = region.centroid
                    coord = [float(centroid[0]) / SHAPE, 
                             float(centroid[1]) / SHAPE]
                    # plot_im(im,centroid)
                    self.coords.append(coord)
        return self.coords 

    def _cal_dist(self, p0):
        print(len(self.coords))
        c0 = self.coords.pop()
        print(c0, p0)
        
    def step(self, action):
        done = False if self.coords else True
        return done
        # self._cal_dist(action)



    def reset(self):
        randidx = np.random.randint(self.size)
        print('randidx:', randidx)
        self.image = self.images[randidx]
        self.coords = self._get_n_masks(self.image)
        pass
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name).unwrapped
env = VoronoiEnvironment(datadir='data', size=100)



if __name__ == '__main__':


    ep_r = 0
    if args.mode == 'random':
        for _ in range(6):
            act = np.random.uniform(0, 1, 2)
            # print(act)
            done = env.step(act)
            print(done)
        pass