# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.quantization import quantize_fx

from module import *

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
    
    def quantize(self, num_bits=8):
        self.qconv = QConv2d(self.conv, qi=True, qo=True, num_bits=num_bits)
        self.qpool = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qbn = QBatchNorm2d(self.bn, qi=False, qo=True, num_bits=num_bits)
        self.qrelu = QReLU()

    def quantize_forward(self, x):
        x = self.qconv(x)
        x = self.qpool(x)
        x = self.qbn(x)
        x = self.qrelu(x)
        return x
    
    def freeze(self):
        self.qconv.freeze()
        self.qpool.freeze(self.qconv.qo)
        self.qbn.freeze(self.qconv.qo)
        self.qrelu.freeze(self.qconv.qo)

    def quantize_inference(self, x):
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx)
        qx = self.qpool.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        qx = self.qrelu.quantize_inference(qx)

    

class non_bottleneck_1d(nn.Module):
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
    
    def quantize(self, num_bits=8):
        self.qconv3x1_1 = QConv2d(self.conv1x3_1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qconv1x3_1 = QConv2d(self.conv3x1_1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qconv3x1_2 = QConv2d(self.conv3x1_2, qi=True, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU()
        self.qconv1x3_2 = QConv2d(self.conv1x3_2, qi=True, qo=True, num_bits=num_bits)
        self.qrelu4 = QReLU()
        self.qdropout2d = QDropout2d(self.dropout, qi=False, qo=True, num_bits=num_bits)
        self.qbn1 = QBatchNorm2d(self.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qbn2 = QBatchNorm2d(self.bn2, qi=False, qo=True, num_bits=num_bits)
    
    def quantize_forward(self, x):
        output = self.qconv3x1_1(x)
        output = self.qrelu1(output)
        output = self.qconv1x3_1(output)
        output = self.qbn1(output)
        output = self.qrelu2(output)

        output = self.qconv3x1_2(output)
        output = self.qrelu3(output)
        output = self.qconv1x3_2(output)
        output = self.qbn2(output)
        output = self.qrelu4(output)

        if hasattr(self, 'qdropout2d'):
            output = self.qdropout2d(output)

        # Apply the identity (residual connection) and ReLU
        output = F.relu(output + x)  # +x = identity (residual connection)
    
        return output

    def freeze(self):
        self.qconv3x1_1.freeze()
        self.qconv1x3_1.freeze()
        self.qrelu1.freeze()
        self.qrelu2.freeze()
        self.qconv3x1_2.freeze()
        self.qconv1x3_2.freeze()
        self.qrelu3.freeze()
        self.qrelu4.freeze()
        self.qdropout2d.freeze()
        self.qbn1.freeze()
        self.qbn2.freeze()



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
        self.output_conv.qconfig = torch.quantization.default_qconfig

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        # output = self.layers[0](output)
        # output = self.layers[1](output)
        # output = self.layers[2](output)
        # output = self.layers[3](output)
        # output = self.layers[4](output)
        # output = self.layers[5](output)
        # output = self.layers[6](output)
        # output = self.layers[7](output)
        # output = self.layers[8](output)
        # output = self.layers[9](output)
        # output = self.layers[10](output)
        # output = self.layers[11](output)
        # output = self.layers[12](output)
        # output = self.layers[13](output)

        if predict:
            output = self.output_conv(output)

        return output
    
    def quantize(self, num_bits=8):
        self.qinitial_block = self.initial_block.quantize()
        for layer in self.layers:
            layer.quantize(num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qinitial_block.quantize_forward(x)
        for layer in self.layers:
            x = layer.quantize_forward(x)
        return x

    def freeze(self):
        self.qinitial_block.freeze()
        for layer in self.layers:
            layer.freeze()

    def quantize_inference(self, x):
        # Quantize the input tensor
        qx = self.qinitial_block.qconv.quantize_tensor(x)
        
        # Pass the quantized input through each layer
        for layer in self.layers:
            qx = layer.quantize_forward(qx)
        
        # Pass the output through the output_conv layer if it exists
        if hasattr(self, 'output_conv'):
            qx = self.output_conv.qconfig(qx)
        
        return qx

    def prepare_fx(self, m, qconfig_dict, example_input):
        self.initial_block = quantize_fx.prepare_fx(self.initial_block, qconfig_dict, example_input)
        l = nn.ModuleList()
        for layer in self. layers:
            layer.qconfig = torch.quantization.default_qconfig
            l.append(quantize_fx.prepare_fx(layer, qconfig_dict, example_input))
        self.layers = nn.ModuleList(l)
        self.output_conv = quantize_fx.prepare_fx(self.output_conv, qconfig_dict, example_input)
        return self




class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.conv.qconfig = torch.quantization.default_qconfig
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

    def quantize(self, num_bits=8):
        self.qconv = QConvTranspose2d(self.conv, qi=True, qo=True, num_bits=num_bits)
        self.qbn = QBatchNorm2d(self.bn, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        output = self.qconv(x)
        output = self.qbn(output)
        return F.relu(output)

    def freeze(self):
        self.qconv.freeze()
        self.qbn.freeze()

    def quantize_inference(self, x):
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        return qx

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
            if hasattr(layer, 'quantize'):
                layer.quantize(num_bits=num_bits)

        if hasattr(self, 'output_conv'):
            self.q_output_conv = QConvTranspose2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        for layer in self.layers:
            if hasattr(layer, 'quantize_forward'):
                x = layer.quantize_forward(x)

        if hasattr(self, 'q_output_conv'):
            x = self.q_output_conv(x)

        return x

    def freeze(self):
        for layer in self.layers:
            if hasattr(layer, 'freeze'):
                layer.freeze()

        if hasattr(self, 'q_output_conv'):
            self.q_output_conv.freeze()

    def quantize_inference(self, x):
        for layer in self.layers:
            if hasattr(layer, 'quantize_inference'):
                x = layer.quantize_inference(x)

        if hasattr(self, 'q_output_conv'):
            x = self.q_output_conv(x)

        return x

    
    def prepare_fx(self, m, qconfig_dict, example_input):
        # l = nn.ModuleList()
        # for layer in self. layers:
        #     layer.qconfig = torch.quantization.default_qconfig
        #     l.append(quantize_fx.prepare_fx(layer, qconfig_dict, example_input))
        # self.layers = l
        # self.output_conv = quantize_fx.prepare_fx(self.output_conv, qconfig_dict, example_input)
        l = nn.ModuleList()
        for layer in self.layers:
            layer.qconfig = torch.quantization.default_qconfig
            l.append(quantize_fx.prepare_fx(layer, qconfig_dict, example_input))
        self.layers = l
        return self

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
        self.qencoder = self.encoder.quantize(num_bits=num_bits)
        self.qdecoder = self.decoder.quantize(num_bits=num_bits)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder.forward(input)    
            return self.decoder.forward(output)

    def quantize_forward(self, x):
        output = self.qencoder.quantize_forward(x)
        return self.qdecoder.quantize_forward(output)

    def freeze(self):
        self.encoder.freeze()
        self.decoder.freeze()

    def quantize_inference(self, x):
        output = self.qencoder.quantize_inference(x)
        return self.qdecoder.quantize_inference(output)

        
    def prepare_fx(self, m, qconfig_dict, example_input):
        self.encoder = self.encoder.prepare_fx(m, qconfig_dict, example_input)
        #print(self.encoder)
        self.decoder = self.decoder.prepare_fx(m, qconfig_dict, example_input)
        #print(self.decoder)
        return self
        
class TestNet(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, groups=20)
        self.fc = nn.Linear(5*5*40, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


