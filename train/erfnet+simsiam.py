# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import os
import gc
import random
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps

import torchvision.transforms as transforms

from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard


from simsiam.builder import SimSiam
import simsiam.loader

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

'''
class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_encoder = models.__dict__['resnet50']
        self.encoder = SimSiam(base_encoder, num_classes)

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output
'''

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.adapt_layer = nn.Sequential(
            # Layer per trasformare [6, 2048] in [6, 128, 4, 4]
            nn.Unflatten(1, (128, 4, 4)),
            # Layer di convoluzione per adattare le dimensioni spaziali
            nn.ConvTranspose2d(128, 128, kernel_size=(16,32), stride=(16,32)),
        ) 

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.adapt_layer(output)
        #print("Output adapt: ", output.shape)
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#ERFNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        base_encoder = models.__dict__['resnet50']
        if (encoder == None):
            #self.encoder = SimSiam(base_encoder, num_classes)
            self.encoder = SimSiam(base_encoder)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def loadInitialWeigth(self,path):
        assert os.path.isfile(path) ,f"loadInitialWeigth path is wrong : {path}"
        if os.path.isfile(path):
            print("Loading weigths SimSiam ... ")
            ckpt = torch.load(path,map_location="cuda" if torch.cuda.is_available() else "cpu")
            ckpt_backbone = {key.replace("module.", ""): value for key, value in ckpt['state_dict'].items()}
            self.encoder.load_state_dict(ckpt_backbone)
            print("Weigths loaded SimSiam ... ")
            del ckpt, ckpt_backbone
            gc.collect()
        else:
            print("NOT Loaded weigths SimSiam ... ")

    def trasform(input, target):
        # hflip = random.random()  # define randomly a value to chose if flip horizontal both images or not (specchiare l'immagine)
        # if (
        #         hflip < 0.5):  # 50% di ruotare l'immagine e 50% no, per aumeentare randomicità nei dati. Per cui alcuni sono specchiati nella fase di augmentation altri no
        #     input = input.transpose(Image.FLIP_LEFT_RIGHT)
        #     target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # transX = random.randint(-2, 2)  # define randomly how much shift the images from 2 pixel to the left to 2 pixel to the right (could be also 0)
        # transY = random.randint(-2, 2)  # define randomly how much shift the images from 2 pixel to the bottom to 2 pixel to the up (could be also 0)

        # input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)  # pad the input è stato riempito con 0 in quei pixel (questo significa che i pixel sono stati resi blu)
        # target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)  # pad label filling with 255 (questo significa che quei pixel sono stati resi bianchi)

        # input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
        # target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))
        # return input, target
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        return simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation))

    def forward(self, input, only_encode=False):
        #input1 = self.trasform(input) #add some trasformations
        #input2 = self.trasform(input) #add some trasformations
        if only_encode:
            return self.encoder.forward(input, input)
        else:
            p1, _, _, _ = self.encoder(input, input)   #predict=False by default
            #print("P1: ", p1.shape)
            return self.decoder.forward(p1)
            #return self.decoder.forward(input)

