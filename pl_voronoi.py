# Author: Tran Minh Quan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#-----------------------------------------------------------------------
import os
import sys
import glob
import random
import shutil
import logging
import argparse
from collections import OrderedDict
from natsort import natsorted

import cv2
import numpy as np
from PIL import Image
import math
import skimage 
#-----------------------------------------------------------------------
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

#-----------------------------------------------------------------------
# # Using tensorflow
# import tensorflow as tf

#-----------------------------------------------------------------------
# An efficient dataflow loading for training and testing
import tensorpack.dataflow as df
from tensorpack import *
import albumentations as AB


from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
# import tqdm

#-----------------------------------------------------------------------
# Global configuration
#
SHAPE = 256
BATCH = 10
EPOCH = 2000
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

class CustomDataFlow(df.DataFlow):
    def __init__(self, size=100, datadir='data/', verbose=True, istrain=True):
        super(CustomDataFlow, self).__init__()
        
        # Manipulate the train and valid set
        if istrain:
            datadir = os.path.join(datadir, 'train')
        else:
            datadir = os.path.join(datadir, 'valid')

        # Read the image
        self.imagedir = os.path.join(datadir, 'images')
        self.imagefiles = natsorted(glob.glob(self.imagedir + '/*.*'))
        self.images = []
        for imagefile in self.imagefiles:
            # print(imagefile)
            image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (SHAPE, SHAPE), cv2.INTER_NEAREST)
            self.images.append(image)
        self.images = np.array(np.squeeze(self.images))

        # Read the label data
        self.labeldir = os.path.join(datadir, 'labels')
        self.labelfiles = natsorted(glob.glob(self.labeldir + '/*.*'))
        self.labels = []
        for labelfile in self.labelfiles:
            # print(labelfile)
            label = cv2.imread(labelfile, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (SHAPE, SHAPE), cv2.INTER_NEAREST)
            self.labels.append(label)
        self.labels = np.array(np.squeeze(self.labels))

        print(self.images.shape)
        print(self.labels.shape)
        self.istrain=istrain
        self.size = size 
        # self.size = min(size, len(self.imagefiles))

    def normal(self, x, shape):
        return (int)(x * (shape - 1) + 0.5)

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            randidx = np.random.randint(self.size)

            # Produce the label
            # Note that we take the position = 255


            if self.istrain: # On the fly image generation
                image = generate_voronoi_diagram(SHAPE, SHAPE, np.random.randint(5, 12))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                image = 255- skimage.segmentation.find_boundaries(image).astype(np.uint8)*255
                label = image.copy()
            else:
                image = np.squeeze(self.images[randidx]).astype(np.uint8)

                label = np.squeeze(self.labels[randidx])
            label = skimage.measure.label(label)
            
            # y0 = 0
            # x0 = 0
            # value = 0 #label[y0, x0]
            
            # while True: #value == 0:
            # while value==0:
            # Produce the distance transform with random location
            y_x_loc = np.random.uniform(0, 1, 2)
            y0, x0 = y_x_loc[0], y_x_loc[1]  
            x0 = self.normal(x0, SHAPE)
            y0 = self.normal(y0, SHAPE)
            value = label[y0, x0]

            field = 255*np.ones([SHAPE, SHAPE]).astype(np.uint8)
            field[y0, x0] = 0
            field = cv2.distanceTransform(field, cv2.DIST_L1, 3)
            field = cv2.normalize(field, field, 0, 255.0, cv2.NORM_MINMAX)
            field = 255-field.astype(np.uint8)

            image = image.astype(np.float32) / 255.0
            field = field.astype(np.float32) / 255.0
            image = image * field

            connc = np.zeros_like(image) # Connected component
            
            if value != 0:
                connc[label==value] = 255
            connc = connc.astype(np.uint8)
            connc = connc.astype(np.float32) / 255.0
            # print(randidx, image.shape, field.shape, connc.shape)


            yield [image.astype(np.float32), 
                   connc.astype(np.float32)]

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

class FusionnetModel(LightningModule):
    def __init__(self, input_nc=1, output_nc=1, ngf=32, ds_train=None, ds_valid=None, ds_test=None):
        super(FusionnetModel,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.ds_test = ds_test
        self.__build_model()


    def __build_model(self):
        print("\n------Initiating FusionNet------\n")
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

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

    

    def forward(self, image):
        down_1 = self.down_1(image)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)
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
        x, y = batch
        y_hat = self.forward(x)
        self.experiment.add_image('train/estim', torch.cat([x, y, y_hat], 3)[0][0], 
                                  self.global_step, dataformats='HW')
        return {'loss': self.dice_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x, y = batch
        y_hat = self.forward(x)
        self.experiment.add_image('valid/estim', torch.cat([x, y, y_hat], 3)[0][0], 
                                  self.global_step, dataformats='HW')
        return {'val_loss': self.dice_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optimizer]

    @pl.data_loader
    def tng_dataloader(self):
        print('tng data loader called')
        return self.ds_train

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.ds_valid

    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.ds_test
    # def fetch_ds_train(self, ds_train):
    #     self.ds_train = ds_train
    
    # def fetch_ds_valid(self, ds_valid):
    #     self.ds_valid = ds_valid

    # def fetch_ds_test(self, ds_test):
    #     self.ds_test = ds_test

    def fetch_dataflow(self, ds_train=None, ds_valid=None, ds_test=None):
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.ds_test = ds_test

#-----------------------------------------------------------------------
# Perform sample
#
# TODO
def sample(datadir='data/', dstdir='result', model=None):
    if model:
        model.eval()
        # Prepare the destination result directory
        shutil.rmtree(dstdir, ignore_errors=True)
        os.makedirs(dstdir)

        imagefiles = natsorted(glob.glob(datadir + '/*.*'))

        for imagefile in imagefiles:
            dstfile = os.path.join(dstdir, os.path.basename(imagefile))
            print(dstfile)
            image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            
            y_x_loc = np.random.uniform(0, 1, 2)
            y0, x0 = y_x_loc[0], y_x_loc[1]  
            x0 = int(SHAPE*x0)
            y0 = int(SHAPE*y0)

            field = 255*np.ones([SHAPE, SHAPE]).astype(np.uint8)
            field[y0, x0] = 0
            field = cv2.distanceTransform(field, cv2.DIST_L1, 3)
            field = cv2.normalize(field, field, 0, 255.0, cv2.NORM_MINMAX)
            field = 255-field.astype(np.uint8)

            image = image.astype(np.float32) / 255.0
            field = field.astype(np.float32) / 255.0
            image = image * field
            cv2.imwrite(dstfile.replace('membr', 'image'), (255*image).astype(np.uint8))
            old_shape = image.shape

            image = cv2.resize(image, (SHAPE, SHAPE))
            image = np.expand_dims(image, 0)
            image = np.expand_dims(image, 1)

            image = torch.tensor(image).float()
            # print(image.max())
            with torch.no_grad():
                estim = model.forward(image).numpy()
            estim = (255.0 * estim).astype(np.uint8)
            estim = np.squeeze(estim)
            estim = cv2.resize(estim, old_shape[::-1])
            print(estim)
            
            cv2.imwrite(dstfile, estim)




#-----------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    #-----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int, help='reproducibility')
    parser.add_argument('--data', default='data', help='the image directory')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--sample', action='store_true', help='run inference')
    args = parser.parse_args()
    print(args)
    #-----------------------------------------------------------------------
    # Choose the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #-----------------------------------------------------------------------
    # Seed the randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    #-----------------------------------------------------------------------
    # Initialize the program
    # TODO
    # writer = SummaryWriter()
    use_cuda = torch.cuda.is_available()
    xpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # step = 0


    #-----------------------------------------------------------------------
    # Train from scratch or load the pretrained network
    #
    # TODO: Create the model
    #-----------------------------------------------------------------------
    print('Create model...')
    model = FusionnetModel(input_nc=1, output_nc=1, ngf=32)
    print('Built model')

    # TODO: Load the pretrained model
    if args.load:
        # TODO
        chkpt = torch.load(args.load, map_location=xpu)
        model.load_state_dict(chkpt['state_dict'])
        print('Loaded from {}...'.format(args.load))

    #-----------------------------------------------------------------------
    # Perform inference
    if args.sample:
        sample(datadir=args.data, model=model)
        sys.exit()
    else:
        #-----------------------------------------------------------------------
        # 1 DATA FLOW
        #-----------------------------------------------------------------------
        ds_train = CustomDataFlow(size=5000, datadir=args.data, istrain=True) #'data)
        ag_image = [
                imgaug.Albumentations(AB.RandomBrightness(limit=0.2, p=0.5)), 
                imgaug.Albumentations(AB.RandomContrast(limit=0.2, p=0.5)), 
                imgaug.Albumentations(AB.GaussianBlur(blur_limit=7, p=0.5)), 
                # imgaug.Albumentations(AB.GaussNoise(var_limit=(10.0, 50.0), p=0.5)), 
                ]
        ag_train = [
                # imgaug.Resize(int(SHAPE * 1.12)),
                # imgaug.RandomCrop(SHAPE),
                # imgaug.Resize(int(SHAPE)),
                imgaug.Flip(horiz=True),
                imgaug.Flip(vert=True),
                imgaug.Albumentations(AB.RandomRotate90(p=1))
                ]
        # ds_train = df.AugmentImageComponent(ds_train, ag_image, 0) # Apply for image only
        ds_train = df.AugmentImageComponents(ds_train, ag_train, (0, 1))
        ds_train = df.MapData(ds_train, lambda dp: [np.expand_dims(dp[0], axis=0), 
                                                    np.expand_dims(dp[1], axis=0), 
                                                    ])

        ds_train = df.MultiProcessRunner(ds_train, num_proc=8, num_prefetch=4)
        ds_train = df.BatchData(ds_train, batch_size=BATCH)
        ds_train = df.PrintData(ds_train)
        ds_train = df.MapData(ds_train, lambda dp: [torch.tensor(dp[0]), 
                                                    torch.tensor(dp[1]), 
                                                    ])
        
        # ds_valid
        ds_valid = CustomDataFlow(size=500, datadir=args.data, istrain=False)
        ag_valid = [
                    imgaug.Flip(horiz=True),
                    imgaug.Flip(vert=True),
                    imgaug.Albumentations(AB.RandomRotate90(p=1))
                    ]
        ds_valid = df.AugmentImageComponents(ds_valid, ag_valid, (0, 1))
        ds_valid = df.MapData(ds_valid, lambda dp: [np.expand_dims(dp[0], axis=0), 
                                                    np.expand_dims(dp[1], axis=0), 
                                                    ])

        ds_valid = df.MultiProcessRunner(ds_valid, num_proc=1, num_prefetch=4)
        ds_valid = df.BatchData(ds_valid, batch_size=5)
        # ds_valid = df.PrintData(ds_valid)
        ds_valid = df.MapData(ds_valid, lambda dp: [torch.tensor(dp[0]), 
                                                    torch.tensor(dp[1]), 
                                                    ])
        #-----------------------------------------------------------------------
        # Attach dataflow to model
        model.fetch_dataflow(ds_train=ds_train,
                           ds_valid=ds_valid, 
                           ds_test=ds_valid)

        #-----------------------------------------------------------------------
        # 2 INIT TEST TUBE EXP
        #-----------------------------------------------------------------------

        # init experiment
        exp = Experiment(
            name='voronoi', #hyperparams.experiment_name,
            save_dir='runs', #hyperparams.test_tube_save_path,
            # autosave=False,
            # description='experiment'
        )

        exp.save()

        #-----------------------------------------------------------------------
        # 3 DEFINE CALLBACKS
        #-----------------------------------------------------------------------
        model_save_path = 'pl_voronoi' #'{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
        early_stop = EarlyStopping(
            monitor='avg_val_loss',
            patience=5,
            verbose=True,
            mode='auto'
        )

        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            # save_best_only=True,
            # save_weights_only=True,
            verbose=True,
            monitor='val_loss',
            mode='auto',
            period=100,
        )

        #-----------------------------------------------------------------------
        # 4 INIT TRAINER
        #-----------------------------------------------------------------------
        trainer = Trainer(
            experiment=exp,
            checkpoint_callback=checkpoint,
            # early_stop_callback=early_stop,
            max_nb_epochs=EPOCH, 
            gpus=args.gpu #map(int, args.gpu.split(',')), #hparams.gpus,
            # distributed_backend='ddp'
        )

        #-----------------------------------------------------------------------
        # 5 START TRAINING
        #-----------------------------------------------------------------------
        trainer.fit(model)
        sys.exit()