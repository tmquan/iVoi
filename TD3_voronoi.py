from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import math
import random
import shutil
import logging
import argparse
from itertools import count
from natsort import natsorted

import cv2
import skimage.io
import skimage.measure
import sklearn
import sklearn.metrics
import numpy as np 
from PIL import Image 
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

#-----------------------------------------------------------------------
# An efficient dataflow loading for training and testing
import tensorpack.dataflow as df
from tensorpack import *
import albumentations as AB


import pytorch_lightning as ptl
from pytorch_lightning import * #Trainer, ModelCheckpoint
from pytorch_lightning.callbacks.pt_callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment
# import tqdm

#-----------------------------------------------------------------------
# Global configuration
#
BATCH = 20
EPOCH = 100000
SHAPE = 256
NF = 64

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
parser.add_argument('--random_seed', default=2021, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--max_length_of_trajectory', default=2000, type=int) # num of games
parser.add_argument('--print_log', default=1, type=int)
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


#-----------------------------------------------------------------------
# Create the model
#
# TODO
# We are going to implement FusionNet
# FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics
# https://arxiv.org/abs/1612.05360 
def conv_block_enc(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_block_dec(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        # nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.Conv2d(in_dim, out_dim*4, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(2), # Subpixel interpolation
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_enc(in_dim,out_dim,act_fn),
        conv_block_enc(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class Conv_block_res(nn.Module):
    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_block_res,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block_enc(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block_enc(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

class FusionnetModel(ptl.LightningModule):
    def __init__(self, input_nc=1, output_nc=1, ngf=32, 
        ds_train=None, ds_valid=None, ds_test=None):
        super(FusionnetModel,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.ds_test = ds_test
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

       

        print("\n------Initiating FusionNet------\n")

        # encoder
        self.down_1 = Conv_block_res(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_block_res(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_block_res(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_block_res(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge
        self.bridge = Conv_block_res(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder
        self.deconv_1 = conv_block_dec(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_block_res(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_block_dec(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_block_res(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_block_dec(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_block_res(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_block_dec(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_block_res(self.out_dim, self.out_dim, act_fn_2)

        # output
        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        # TODO: add graph here

    def forward(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
        #out = torch.clamp(out, min=-1, max=1)
        out = (out + 1.0) / 2.0
        return out

    def dice_loss(self, input, target):
        smooth=.001
        input=input.view(-1)
        target=target.view(-1)
        return(1-2*(input*target).sum()/(input.sum()+target.sum()+smooth))

    def training_step(self, batch, batch_nb):
        x, f, y = batch
        y_hat = self.forward(torch.cat([x, f], 1))
        self.experiment.add_image('train/estim', torch.cat([x, f, y, y_hat], 3)[0][0], 
                                  self.global_step, dataformats='HW')
        return {'loss': self.dice_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, f, y = batch
        y_hat = self.forward(torch.cat([x, f], 1))
        self.experiment.add_image('valid/estim', torch.cat([x, f, y, y_hat], 3)[0][0], 
                                  self.global_step, dataformats='HW')
        return {'val_loss': self.dice_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-4)]

    @ptl.data_loader
    def tng_dataloader(self):
        return self.ds_train

    @ptl.data_loader
    def val_dataloader(self):
        return self.ds_valid

    @ptl.data_loader
    def test_dataloader(self):
        return self.ds_test


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

#-----------------------------------------------------------------------
# Create a dataflow with tensorpack dataflow 
# independent to tf and pytorch, we can use pure python to do this
# TODO
def generate_voronoi_diagram(width, height, num_cells):
    image = Image.new("RGB", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    for i in range(num_cells):
        nx.append(np.random.randint(imgx))
        ny.append(np.random.randint(imgy))
        nr.append(np.random.randint(256))
        ng.append(np.random.randint(256))
        nb.append(np.random.randint(256))
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j]))
    return np.array(image)


class VoronoiEnvironment(CustomEnvironment):
    def __init__(self, datadir='data/voronoi', size=100, istrain=True, writer=None, ckpt=None):
        self.size = size
        self.ckpt = ckpt
        self._load_images(datadir=datadir)
        self.reset()
        self.fusionnet = FusionnetModel(input_nc=1, output_nc=1, ngf=32)
        self._load_pretrained(self.ckpt)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.global_step = 0
        self.writer= writer
    def _load_pretrained(self, ckpt=None):
        
        # use_cuda = torch.cuda.is_available()
        xpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt:
            ckpt = torch.load(self.ckpt, map_location=xpu)
            self.fusionnet.load_state_dict(ckpt['state_dict'])
            print('Loaded from {}...'.format(self.ckpt))

        else:
            return None
     
    def _load_images(self, datadir=None):
        self.datadir = datadir
        
        # Query the image
        self.imagedir = os.path.join(datadir, 'train')
        self.imagefiles = natsorted(glob.glob(self.imagedir + '/*.*'))
        
        self.size = min(self.size, len(self.imagefiles))

        self.imagefiles = self.imagefiles[:self.size]
        self.images = []
        for imagefile in self.imagefiles:
            # print(imagefile)
            image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            # image = cv2.resize(image, (SHAPE, SHAPE))
            self.images.append(image)
        self.images = np.array(np.squeeze(self.images))

    def reset(self):
        # randidx = np.random.randint(self.size)
        # print('randidx:', randidx)
        # self.image = self.images[randidx]
        self.image = generate_voronoi_diagram(SHAPE, SHAPE, np.random.randint(5, 12))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.image = 255- skimage.segmentation.find_boundaries(self.image).astype(np.uint8)*255

        image = self.image.copy()
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 1) # 1 1 w h 

        self.label = skimage.measure.label(self.image).astype(np.int32)
        label = self.label.copy()
        label = np.expand_dims(label, 0)
        label = np.expand_dims(label, 1)

        self.coords = self._get_n_masks(self.label)
        self.zeros = np.zeros_like(self.image)
        zeros = self.zeros.copy()
        zeros = np.expand_dims(zeros, 0)
        zeros = np.expand_dims(zeros, 1) # 1 1 w h 

        self.state = np.zeros_like(self.image)
        state = self.state.copy()
        state = np.expand_dims(state, 0)
        state = np.expand_dims(state, 1) # 1 1 w h 
        self.order = 0
        
        obs = np.concatenate([image / 255.0, zeros / 255.0, zeros / 255.0], 1)

        curr_dist = 1-sklearn.metrics.adjusted_rand_score(self.label.flatten(), self.state.flatten())
        self.prev_dist = curr_dist
        self.viz = []
        return obs
        
    def _get_n_masks(self, label):
        self.coords = []
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
       
    def step(self, action):
        # Need to return obs, rwd, done, info 
        obs = None 
        rwd = None 
        done = None
        info = None
        
        y0, x0 = int(action[0]*SHAPE), int(action[1]*SHAPE)
        seedval = self.label[y0, x0]

        self.field = 255*np.ones([SHAPE, SHAPE]).astype(np.uint8)
        self.field[y0, x0] = 0
        self.field = cv2.distanceTransform(self.field, cv2.DIST_L1, 3)
        self.field = cv2.normalize(self.field, self.field, 0, 255.0, cv2.NORM_MINMAX)
        self.field = 255-self.field.astype(np.float32)
        

        connc = np.zeros_like(self.label, dtype=np.float32) # Connected component
        # print(self.label.shape, connc.shape)
        connc[self.label==seedval] = 1
        connc = connc.astype(np.float32)
        
        # Run thru the network
        image = self.image.copy()
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 1) # 1 1 w h 
        
        field = self.field.copy()
        field = np.expand_dims(field, 0)
        field = np.expand_dims(field, 1) # 1 1 w h 

        
        with torch.no_grad():
            estim = 255 * self.fusionnet(torch.tensor(image / 255.0 * field / 255.0 ).float()).detach().numpy() #.astype(np.float32)
        self.estim = np.squeeze(estim)
      
        if args.mode=='random':
            cv2.imwrite('image_{}_{}.png'.format(y0, x0), np.squeeze(255*(image*field).detach().numpy()).astype(np.uint8))
            cv2.imwrite('estim_{}_{}.png'.format(y0, x0), np.squeeze(255*estim))
  
        

        # We have estim, now we need to calculate the observation base on the estim
        self.order = self.order+1
        self.last_state = self.state.copy()
        self.state = self.order*(self.estim>128) + self.last_state # similar to label
        state = self.state.copy()
        state = np.expand_dims(state, 0)
        state = np.expand_dims(state, 1) # 1 1 w h 


        # self.proba = 255*(self.state > 0)
        self.proba = 255- skimage.segmentation.find_boundaries(self.last_state).astype(np.uint8)*255
        self.proba[self.state==0] = 0
        proba = self.proba.copy()
        proba = np.expand_dims(proba, 0)
        proba = np.expand_dims(proba, 1) # 1 1 w h 

        # print(image.shape, estim.shape, proba.shape)
        obs = np.concatenate([image / 255.0, estim / 255.0,  proba / 255.0], 1)

         # Calculate the done
        num_proba_pixels = np.sum(self.proba > 128)
        per_proba_pixels = num_proba_pixels / (SHAPE * SHAPE)
        # print(len(self.coords), per_proba_pixels)
        # done = False if self.coords or per_proba_pixels < 0.95 else True
        done = True if len(self.coords)==0 else False #or per_proba_pixels > 0.95 else False 

        #####################################################################
        # # Calculate the reward
        # # Pick the closest point in the list and pop that from the coords
        # # min(myList, key=lambda x:abs(x-myNumber))

        def cal_l2_dist(pointA, pointB):
            dist = np.linalg.norm(pointA - pointB) / np.sqrt(2)
            return dist

        if not done:
            popped = min(self.coords, key=lambda x: cal_l2_dist(x, action))
            # print('-'*50)
            # print('action:', (action))
            # print('coords:', (self.coords))
            # print('popped:', popped)
            # print('cal_l2_dist:', cal_l2_dist(popped, action))
            self.coords.remove(popped)

            # Calculate 2 rewards: distance rewards and dice reward
            norm_rwd = 1-cal_l2_dist(popped, action) 
        else:
            norm_rwd = 1

        # # seedings = [int32(action[0] * SHAPE), 
        # #             int32(action[1] * SHAPE)]
        # corrects = np.zeros_like(self.label)
        # corrects[self.label==(self.label[y0, x0])] = 255.0

        def dice_dist(estim, connc):
            # print(estim, connc)
            smooth = .001
            estim = estim.flatten()
            connc = connc.flatten()
            dist = (2*(estim*connc).sum()/(estim.sum()+connc.sum()+smooth))
            return dist

        # dice_rwd = 1-dice_dist(estim / 255.0, corrects / 255.0)

        

        # if done:
        #     # sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)[source]
        #     rand_rwd = sklearn.metrics.adjusted_rand_score(self.label.flatten(), self.state.flatten())
        # else:
        #     rand_rwd = 0

        # rwd = (norm_rwd * dice_rwd + 100*rand_rwd)
        curr_dist = (1-sklearn.metrics.cluster.fowlkes_mallows_score(self.label.flatten(), self.state.flatten())) * dice_dist(self.estim.flatten() / 255.0, self.proba.flatten() / 255.0) 
        rwd = (self.prev_dist - curr_dist) * norm_rwd
        





        # Calculate the additional info
        info = None 
        # print('norm_rwd: {:0.5f}, dice_rwd: {:0.5f}, rand_rwd: {:0.5f}, step_rwd {:0.5f}, num_obj: {:02d}, done: {}, info {}'
        #     .format(norm_rwd, dice_rwd, rand_rwd, rwd, len(self.coords), done, info))
        print('prev_dist: {:0.5f}, curr_dist: {:0.5f}, rwd: {:0.5f}, norm_rwd: {:0.5f}, num_obj: {:02d}, done: {}, info {}'
                .format(self.prev_dist, curr_dist, rwd, norm_rwd, len(self.coords), done, info))

        # Log to tensorboard
        # print(self.image.shape, self.label.shape, self.membr.shape, self.field.shape, self.estim.shape, self.state.shape)
        self.viz.append(np.concatenate([self.image / 255.0, 
                                                 self.label / (np.max(self.label) + 1e-6), 
                              
                                                 self.image / 255.0 * self.field / 255.0, 
                                                 self.estim / 255.0, 
                                                 self.proba / 255.0,
                                                 self.state / (np.max(self.state) + 1e-6),
                                                 ], 1)) #[0][0], )

        self.writer.add_image('step', np.concatenate(self.viz, 0),
                                  self.global_step, dataformats='HW')

        # Any update on the step
        self.global_step += 1
        self.prev_dist = curr_dist
        return obs, rwd, done, info






device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name).unwrapped

writer = SummaryWriter()
env = VoronoiEnvironment(datadir='data/voronoi', size=100, 
                         ckpt='pl_voronoi/_ckpt_epoch_1000.ckpt', 
                         writer=writer)


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


###################################################################################
# Resnet definition
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Actor(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(Actor, self).__init__(block, layers, num_classes)
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, action_dim)
#         self.max_action = max_action

#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.max_action * torch.tanh(self.l3(x))
#         return x

class Critic(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(Critic, self).__init__(block, layers, num_classes)
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         self.l1 = nn.Linear(state_dim + action_dim, 400)
#         self.l2 = nn.Linear(400 , 300)
#         self.l3 = nn.Linear(300, 1)

#     def forward(self, x, u):
#         x = F.relu(self.l1(torch.cat([x, u], 1)))
#         x = F.relu(self.l2(x))
#         x = self.l3(x)
#         return x




###################################################################################
class TD3():
    def __init__(self, writer=None):

        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        # self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        # self.critic_1 = Critic(state_dim, action_dim).to(device)
        # self.critic_1_target = Critic(state_dim, action_dim).to(device)
        # self.critic_2 = Critic(state_dim, action_dim).to(device)
        # self.critic_2_target = Critic(state_dim, action_dim).to(device)

        # self.actor_optimizer = optim.Adam(self.actor.parameters())
        # self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        # self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        # self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor = Actor(Bottleneck, [3, 4, 6, 3], num_classes=2).to(device)
        self.actor_target = Actor(Bottleneck, [3, 4, 6, 3], num_classes=2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.learning_rate)

        self.critic_1 = Critic(Bottleneck, [3, 4, 6, 3], num_classes=1).to(device)
        self.critic_1_target = Critic(Bottleneck, [3, 4, 6, 3], num_classes=1).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), args.learning_rate)

        self.critic_2 = Critic(Bottleneck, [3, 4, 6, 3], num_classes=1).to(device)
        self.critic_2_target = Critic(Bottleneck, [3, 4, 6, 3], num_classes=1).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), args.learning_rate)

        # self.max_action = max_action
        # self.memory = Replay_buffer(args.capacity)
        self.memory = Replay_buffer(args.capacity)
        self.writer = writer if writer else SummaryWriter()
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state).to(device)
        # print(state.shape)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.actor_target.state_dict(), 'actor_target.pth')
        torch.save(self.critic_1.state_dict(), 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.actor_target.load_state_dict(torch.load('actor_target.pth'))
        self.critic_1.load_state_dict(torch.load('critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load('critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load('critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load('critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = TD3(writer=writer)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        for i in range(args.max_episode):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)

                # issue 3 add noise to action
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                ep_r += reward
                if args.render and i >= args.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float(done)))
                # if (i+1) % 10 == 0:
                #     print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))

                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % args.log_interval == 0:
                agent.save()
            if len(agent.memory.storage) >= args.capacity-1:
                agent.update()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    if args.mode == 'random':
        for _ in range(3):
            act = np.random.uniform(0, 1, 2)
            # print(act)
            obs, rwd, done, info = env.step(act)
            print(done)
    else:
        main()