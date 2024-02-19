# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy
# from compute_flops import compute_flops

from module import *

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = input
        inputMaxPool = output
        if hasattr(self, "adaptingInput") and self.conv.in_channels + self.conv.out_channels != self.bn.num_features:
            inputMaxPool = self.adaptingInput(output)
        output = torch.cat([self.conv(output), self.pool(inputMaxPool)], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output

    def quantize(self, num_bits=8):
        return
        # self.qconv
        # self.pool = QMaxP(self.conv, self.pool, qi=True, qo=True, num_bits=num_bits)
        # self.qbn = QBN(self.bn, qi=False, qo=True)
        # self.qrelu = QReLU()

    def quantize_forward(self, input): ##calibrazione
        return self.forward(input)
        # x = self.qconvpool(input)
        # x = self.qbn(x)
        # x = self.qrelu(x)
        # return x

    def freeze(self): ##salvo qi e qo
        return
        # self.qconvpool.freeze()
        # self.qbn.freeze(qi=self.qconvpool.qo)
        # self.qrelu.freeze(self.qbn.qo)

    def quantize_inference(self, x):
        return self.forward(x)
        # qx = self.qconvpool.qi.quantize_tensor(x)
        # qx = self.qconvpool.quantize_inference(qx)
        # qx = self.qbn.quantize_inference(qx)
        # qx = self.qrelu.quantize_inference(qx)
        # out = self.qrelu.qi.dequantize_tensor(qx)
        # return out


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.relu1 = nn.ReLU()
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.relu2 = nn.ReLU()
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.relu3 = nn.ReLU()
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.relu4 = nn.ReLU()
        
    def forward(self, input):

        output = self.conv3x1_1(input)
        output = self.relu1(output)

        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu2(output)

        output = self.conv3x1_2(output)
        output = self.relu3(output)
        
        output = self.conv1x3_2(output)
        output = self.bn2(output)


        if hasattr(self, 'adaptingInput') and input.size()[1] != output.size()[1]:
            input = self.adaptingInput(input)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = self.relu4(output+input)    #+input = identity (residual connection)
        return output

    def quantize(self, num_bits=8):
        # self.qblock = QuantizedBN1d(self.conv1x3_1,
        #                             self.conv1x3_1, self.bn1,
        #                             self.conv3x1_2,
        #                             self.conv1x3_2, self.bn2)
        # self.qblock.qconfig = torch.ao.quantization.get_default_qconfig(confg)
        # self.qblock.eval()
        # self.qblock = torch.ao.quantization.fuse_modules(self.qblock, [
        #     ['conv3x1_1', 'relu1'],
        #     ['conv1x3_1', 'bn1', 'relu2'],
        #     ['conv3x1_2', 'relu3'],
        #     ['conv1x3_2', 'bn2']
        #     ])
        # self.qblock = torch.ao.quantization.prepare(self.qblock)
        # return
        self.qconv3x1_1 = QConv2d(copy.deepcopy(self.conv3x1_1), qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qconv1x3_1 = QConvBNReLU(copy.deepcopy(self.conv1x3_1), self.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv3x1_2 = QConv2d(copy.deepcopy(self.conv3x1_2), qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU()
        self.qconv1x3_2 = QConv2d(copy.deepcopy(self.conv1x3_2), qi=False, qo=True, num_bits=num_bits)

        del self.conv3x1_1
        del self.relu1
        del self.conv1x3_1
        del self.bn1
        del self.relu2
        del self.conv3x1_2
        del self.relu3
        del self.conv1x3_2
        del self.dropout
        # self.qbn2 = QBN(self.bn2, qi=False, qo=True)
        # self.qrelu3 = QReLU()

    def quantize_forward(self, input):
        # output = self.qblock(input)
        # if hasattr(self, 'adaptingInput') and input.size()[1] != output.size()[1]:
        #     input = self.adaptingInput(input)

        # output = self.relu4(output+input)    #+input = identity (residual connection)
        # return output
        output = self.qconv3x1_1(input)
        output = self.qrelu1(output)
        
        output = self.qconv1x3_1(output)
        
        output = self.qconv3x1_2(output)
        output = self.qrelu3(output)
        
        output = self.qconv1x3_2(output)

        output = self.bn2(output)

        if hasattr(self, 'adaptingInput') and input.size()[1] != output.size()[1]:
            input = self.adaptingInput(input)

        output = self.relu4(output + input)  # +x = identity (residual connection)
        return output

    def freeze(self):
        # self.qblock = torch.ao.quantization.convert(self.qblock)
        # return
        # print("##### Freezing NB1D #####")
        self.qconv3x1_1.freeze()
        self.qrelu1.freeze(self.qconv3x1_1.qo)
        self.qconv1x3_1.freeze(qi=self.qconv3x1_1.qo)
        self.qconv3x1_2.freeze(qi=self.qconv1x3_1.qo)
        self.qrelu3.freeze(self.qconv3x1_2.qo)
        self.qconv1x3_2.freeze(qi=self.qconv3x1_2.qo)
        # if hasattr(self, 'qdropout2d'):
        #     self.qdropout2d.freeze(self.qconv1x3_2.qo)
        # self.qbn2.freeze(qi=self.qconv1x3_2.qo)
        # self.qrelu3.freeze(self.qbn2.qo)
        # self.qrelu3.freeze(self.qconv3x1_2.qo)

    def quantize_inference(self, x):
        # output = self.qblock(x)
        # if hasattr(self, 'adaptingInput') and x.size()[1] != output.size()[1]:
        #     x = self.adaptingInput(x)

        # if (self.dropout.p != 0):
        #     output = self.dropout(output)

        # output = self.relu4(output+x)    #+input = identity (residual connection)
        # return output
        qx = self.qconv3x1_1.qi.quantize_tensor(x)
        qx = self.qconv3x1_1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qconv1x3_1.quantize_inference(qx)
        qx = self.qconv3x1_2.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(qx)
        qx = self.qconv1x3_2.quantize_inference(qx)
        qx = self.qconv1x3_2.qo.dequantize_tensor(qx)

        qx = self.bn2(qx)
        
        if hasattr(self, 'adaptingInput') and x.size()[1] != qx.size()[1]:
            x = self.adaptingInput(x)
        out = self.relu3(qx+x)
        # out = self.relu3.qi.dequantize_tensor(qx)
        return out

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        
        for layer in self.layers:
            output = layer(output)
            
            
        if predict:
            output = self.output_conv(output)
        
        return output


    def quantize(self, num_bits=8):
        self.initial_block.quantize(num_bits)
        for layer in self.layers:
            layer.quantize(num_bits)
        # self.qoutput_conv = QConv2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.initial_block.quantize_forward(x)
        
        for layer in self.layers:
            x = layer.quantize_forward(x)
       
        return x

    def freeze(self):
        self.initial_block.freeze()
        for layer in self.layers:
            layer.freeze()

    def quantize_inference(self, x):
        qx = self.initial_block.quantize_inference(x)

        for layer in self.layers:
            qx = layer.quantize_inference(qx)
            
        return qx
    


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output =  self.relu(output)
        return output

    def quantize(self, num_bits=8):
        self.qconv = QConvTBNReLU(copy.deepcopy(self.conv), copy.deepcopy(self.bn), qi=True, qo=True, num_bits=num_bits)
        del self.conv
        del self.relu
        del self.bn

    def quantize_forward(self, input):
        output = self.qconv(input)
        return output

    def freeze(self):
        self.qconv.freeze()
        
    def quantize_inference(self, x):
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx)
        out = self.qconv.qo.dequantize_tensor(qx)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    
    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)
        
        output = self.output_conv(output)

        return output

    def quantize(self, num_bits=8):
        for layer in self.layers:
            layer.quantize(num_bits)

        self.q_output_conv = QConvTranspose2d(self.output_conv, qi=True, qo=True, num_bits=8)
        
    
    def quantize_forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.quantize_forward(output)

        output = self.q_output_conv(output)

        return output

    def freeze(self):
        for layer in self.layers:
            layer.freeze()

        self.q_output_conv.freeze()

    def quantize_inference(self, x):
        qx = x
        for layer in self.layers:
            if hasattr(layer, 'quantize_inference'):
                qx = layer.quantize_inference(qx)
        qx = self.q_output_conv.qi.quantize_tensor(qx)
        qx = self.q_output_conv.quantize_inference(qx)
        out = self.q_output_conv.qo.dequantize_tensor(qx)

        return out
    


#ERFNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        

    def quantize(self, num_bits=8):
        self.encoder.quantize(num_bits)
        self.decoder.quantize(num_bits)
        
        # qencoder_params = list(self.encoder.parameters())
        # self.register_parameter('qencoder', nn.Parameter(qencoder_params))
        # self.register_parameter('qdecoder', nn.Parameter(self.encoder.parameters()))
            
    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder.forward(input)
            return self.decoder.forward(output)

    def quantize_forward(self, x):
        # print("Input: ", x.shape)
        output = self.encoder.quantize_forward(x)
        # print("Output: ", x.shape)
        return self.decoder.quantize_forward(output)
        # return self.decoder.quantize_forward(output)

    def freeze(self):
        self.encoder.freeze()
        self.decoder.freeze()

    def quantize_inference(self, x):
        # print("### Encoder input: ", x.shape)
        output = self.encoder.quantize_inference(x)
        # output = self.encoder.quantize_inference(x)
        # output = self.decoder.forward(output)
        output = self.decoder.quantize_inference(output)
        return output
