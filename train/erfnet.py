# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from compute_flops import compute_flops

from module import *

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # print("In: {}, out: {}", ninput, noutput)
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        #print("#### DownSampler ####")
        #print("Input: ", input.shape)
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        #print("Conv+pool: ", output.shape)
        # output = self.conv(input)
        # print("Pool: ", input.shape)
        # output = self.pool(input)
        output = self.bn(output)
        #print("BN: ", output.shape)
        output = F.relu(output)
        #print("Relu: ", output.shape)
        #print("Output: ", output.shape)
        #print("#### END ####")
        return output

    def quantize(self, num_bits=8):
        self.qconv = QConv2d(self.conv, qi=True, qo=True, num_bits=num_bits)
        self.qpool = QMaxPooling2d(kernel_size=2, stride=2, padding=0, qi=True, num_bits=num_bits)
        self.qrelu = QReLU()
        # self.qconv = QConvMaxPoolBN2d(self.conv, self.pool, self.bn, qi=True, qo=True, num_bits=num_bits)
        # self.qrelu = QReLU()
        #oppure
        # self.qconv = QConvMaxPoolBNReLU(self.conv, self.pool, self.bn, qi=True, qo=True, num_bits=num_bits)
        return self

    def quantize_forward(self, input):
        # print("#### QDS ####")
        #print("Input: ", input.shape)
        # x = self.qconv(x)
        # print("QPool_f: ", x.shape)
        # x = self.qpool(x)
        x = torch.cat([self.qconv(input), self.qpool(input)], 1)
        #print("QConv+QPool_f: ", x.shape)
        # print("BN: ", x.shape)
        # x = self.bn(x)
        x = self.qrelu(x)
        #print("Qrelu_f: ", x.shape)
        #print("Output: ", x.shape)
        # print(x.shape)
        # x = self.qrelu(x)
        #print("#### END ####")
        return x

    def freeze(self):
        # print("##### Freezing DS #####")
        self.qconv.freeze()
        self.qpool.freeze()
        self.qrelu.freeze(self.qconv.qo)
        # self.qrelu.freeze(self.qconv.qo)

    def quantize_inference(self, x):
        # print("#### QI DS ####")
        # print("Input: ", x.shape)
        qx = self.qconv.qi.quantize_tensor(x)
        # qx_p = self.qpool.qi.quantize_tensor(qx)
        # print("Cat: ", qx.shape)
        # print("Cat p: ", qx_p.shape)
        qx = torch.cat([self.qconv.quantize_inference(qx), self.qpool.quantize_inference(qx)], 1)
        # print("Cat: ", qx.shape)
        # qx = self.qrelu.quantize_inference(qx)
        out = self.qrelu.quantize_inference(qx)
        # print("Output: ", qx.shape)
        # out = self.qconv.qo.dequantize_tensor(qx)
        return out

    def get_flops_and_params(self):
        print("### DS ###")
        params = sum(p.numel() for p in self.qconv.parameters() if p.requires_grad)
        # print("params: ", params)
        # params += sum(p.numel() for p in self.qpool.parameters() if p.requires_grad)
        # print("params: ", params)
        # params += sum(p.numel() for p in self.qrelu.parameters() if p.requires_grad)
        print("### total params: {} ###".format(params))
        return 1, params


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
        #print("#### NB1D ####")
        #print("Input: ", input.shape)
        output = self.conv3x1_1(input)
        # print("Conv: ", output.shape)
        output = F.relu(output)
        # print("Relu: ", output.shape)

        output = self.conv1x3_1(output)
        # print("Conv: ", output.shape)
        output = self.bn1(output)
        # print("BN: ", output.shape)
        output = F.relu(output)
        # print("Relu: ", output.shape)

        output = self.conv3x1_2(output)
        # print("Conv: ", output.shape)
        output = F.relu(output)
        # print("Relu: ", output.shape)
        output = self.conv1x3_2(output)
        # print("Conv: ", output.shape)
        output = self.bn2(output)
        # print("BN: ", output.shape)

        if (self.dropout.p != 0):
            output = self.dropout(output)
            #print("Drop: ", output.shape)

        output = F.relu(output+input)    #+input = identity (residual connection)
        # print("Relu: ", output.shape)
        #print("Output: ", output.shape)
        #print("#### END ####")
        return output
        #return F.relu(output+input)    #+input = identity (residual connection)

    def quantize(self, num_bits=8):
        self.qconv3x1_1 = QConv2d(self.conv1x3_1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qconv1x3_1 = QConvBNReLU(self.conv3x1_1, self.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv3x1_2 = QConv2d(self.conv3x1_2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qconv1x3_2 = QConv2d(self.conv1x3_2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU()
        # self.qconv3x1_2 = QConvBNReLU(self.conv3x1_2, self.bn2, qi=True, qo=True, num_bits=num_bits)
        return self

    def quantize_forward(self, input):
        # print("### QNB1D ###")
        # print("Input: ", input.shape)
        output = self.qconv3x1_1(input)
        # print("QConv3x1_1_f: ", output.shape)
        output = self.qrelu1(output)
        # print("QRelu1_f: {}", output.shape)
        output = self.qconv1x3_1(output)
        # print("QConv1x3_1_f: ", output.shape)
        
        output = self.qconv3x1_2(output)
        # print("QConv3x1_2_f: ", output.shape)
        output = self.qrelu2(output)
        # print("QRelu2f: ", output.shape)
        
        output = self.qconv1x3_2(output)
        # print("QConv1x3_2_f: ", output.shape)
        # output = self.qbn2(output)

        if hasattr(self, 'qdropout2d'):
            output = self.qdropout2d(output)
            #print("QDropout_f: ", output.shape)

        # Apply the identity (residual connection) and ReLU
        output = self.qrelu3(output + input)  # +x = identity (residual connection)
        # print("QRelu3_f: {}", output.shape)
        #print("Outpuy: ", output.shape)
        #print("#### END ####")
        return output

    def freeze(self):
        # print("##### Freezing NB1D #####")
        self.qconv3x1_1.freeze()
        self.qrelu1.freeze(self.qconv3x1_1.qo)
        self.qconv1x3_1.freeze(qi=self.qconv3x1_1.qo)
        self.qconv3x1_2.freeze(qi=self.qconv1x3_1.qo)
        self.qrelu2.freeze(self.qconv3x1_2.qo)
        self.qconv1x3_2.freeze(qi=self.qconv3x1_2.qo)
        # if hasattr(self, 'qdropout2d'):
        #     self.qdropout2d.freeze(self.qconv1x3_2.qo)
        self.qrelu3.freeze(self.qconv1x3_2.qo)
        # self.qbn2.freeze()

    def quantize_inference(self, x):
        # print("#### QI NB1D ####")
        # print("Input: ", x.shape)
        qx = self.qconv3x1_1.qi.quantize_tensor(x)
        # print("Con: ", qx.shape)
        qx = self.qconv3x1_1.quantize_inference(qx)
        # print("Con: ", qx.shape)
        qx = self.qrelu1.quantize_inference(qx)
        # print("Relu: ", qx.shape)
        qx = self.qconv1x3_1.quantize_inference(qx)
        # print("Conv: ", qx.shape)
        qx = self.qconv3x1_2.quantize_inference(qx)
        # print("Conv: ", qx.shape)
        qx = self.qrelu2.quantize_inference(qx)
        # print("Relu: ", qx.shape)
        qx = self.qconv1x3_2.quantize_inference(qx)
        if hasattr(self, 'qdropout2d'):
            qx = self.quantize_inference(qx)
        # print("Conv: ", qx.shape)
        out = self.qrelu3.quantize_inference(qx)
        # print("Output: ", out.shape)
        return out
    
    def get_flops_and_params(self):
        print("### NB1D ###")
        params = sum(p.numel() for p in self.qconv3x1_1.parameters() if p.requires_grad)
        print("params: ", params)
        # params += sum(p.numel() for p in self.qrelu1.parameters() if p.requires_grad)
        # print("params: ", params)
        params += sum(p.numel() for p in self.qconv1x3_1.parameters() if p.requires_grad)
        print("params: ", params)
        params += sum(p.numel() for p in self.qconv3x1_2.parameters() if p.requires_grad)
        print("params: ", params)
        # params += sum(p.numel() for p in self.qrelu2.parameters() if p.requires_grad)
        # print("params: ", params)
        params += sum(p.numel() for p in self.qconv1x3_2.parameters() if p.requires_grad)
        # print("params: ", params)
        # params += sum(p.numel() for p in self.qrelu3.parameters() if p.requires_grad)
        print("### Total params: {} ###".format(params))
        return 1, params


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
        # self.output_conv.qconfig = torch.quantization.default_qconfig

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

    def quantize(self, num_bits=8):
        self.qinitial_block = self.initial_block.quantize()
        self.qlayers = nn.ModuleList()
        for layer in self.layers:
            self.qlayers.append(layer.quantize(num_bits=num_bits))
        self.qoutput_conv = QConv2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)
        return self

    def quantize_forward(self, x, predict=False):
        # print("### Encoder (predict {}) ###".format(predict))
        x = self.qinitial_block.quantize_forward(x)
        
        for layer in self.qlayers:
            x = layer.quantize_forward(x)
        if predict:
            # print("### Output Conv ###")
            x = self.qoutput_conv(x)
        return x

    def freeze(self, predict=False):
        # print("#### Freezing Encoder ####")
        self.qinitial_block.freeze()
        for layer in self.qlayers:
            layer.freeze()
        if predict:
            self.qoutput_conv.freeze()

    def quantize_inference(self, x):
        # print("#### Quantize Inference Encoder ####")
        qx = self.qinitial_block.quantize_inference(x)

        # Pass the quantized input through each layer
        for layer in self.qlayers:
            qx = layer.quantize_inference(qx)

        # Pass the output through the output_conv layer if it exists
        # qx = self.qoutput_conv.quantize_inference(qx)
        return qx
    
    def get_flops_and_params(self):
        flops, params = self.qinitial_block.get_flops_and_params()
        # print(params)
        for l in self.layers:
            flops_l, params_l = l.get_flops_and_params()
            params += params_l
            flops += flops_l
        # param = sum(p.numel() for p in self.qoutput_conv.parameters())
        # params += param
        return flops, params


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.conv.qconfig = torch.quantization.default_qconfig
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        # print("#### US ####")
        # print("Input: ", input.shape)
        output = self.conv(input)
        # print("Conv: ", output.shape)
        output = self.bn(output)
        # print("Bn: ", output.shape)
        output =  F.relu(output)
        # print("Relu: ", output.shape)
        # print("Output: ", output.shape)
        return output
        # return F.relu(output)

    def quantize(self, num_bits=8):
        self.qconv = QConvTBNReLU(self.conv, self.bn, qi=True, qo=True, num_bits=num_bits)
        # self.qbn = self.bn
        return self

    def quantize_forward(self, input):
        # print("#### QUS ####")
        # print("Input: ", input.shape)
        output = self.qconv(input)
        # output = self.qbn(output)
        # return F.relu(output)
        return output

    def freeze(self):
        # print("##### Freezing US #####")
        self.qconv.freeze()
        # self.qbn.freeze()

    def quantize_inference(self, x):
        # print("#### QI US ####")
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx)
        out = self.qconv.qo.dequantize_tensor(qx)
        return out
    
    def get_flops_and_params(self):
        params = sum(p.numel() for p in self.qconv.parameters() if p.requires_grad)
        print("### US params: {} ###".format(params))
        return 0, params

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

        self.q_output_conv = QConvTranspose2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)
        return self
    
    def quantize_forward(self, input):
        # print("### Decoder ###")
        output = input
        for layer in self.layers:
            output = layer.quantize_forward(output)

        output = self.q_output_conv(output)

        return output

    def freeze(self):
        # print("#### Freezing Decoder ####")
        for layer in self.layers:
            layer.freeze()

        self.q_output_conv.freeze()

    def quantize_inference(self, x):
        # print("#### Quantize Inference Decoder ####")
        for layer in self.layers:
            if hasattr(layer, 'quantize_inference'):
                x = layer.quantize_inference(x)

        if hasattr(self, 'q_output_conv'):
            x = self.q_output_conv(x)

        return x
    
    def get_flops_and_params(self):
        flops = 0
        params = 0
        for l in self.layers:
            flops_l, params_l = l.get_flops_and_params()
            flops += flops_l
            params += params_l
        for name, param in self.q_output_conv.named_parameters():
            if param.requires_grad:
                print(name)
        params_o = sum(p.numel() for p in self.q_output_conv.parameters() if p.requires_grad)
        params += params_o
        return flops, params


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
        output = self.qencoder.quantize_forward(x)
        # print("Output: ", x.shape)
        return self.qdecoder.quantize_forward(output)

    def freeze(self):
        self.encoder.freeze()
        self.decoder.freeze()

    def quantize_inference(self, x):
        # print("### Encoder input: ", x.shape)
        output = self.qencoder.quantize_inference(x)
        return self.qdecoder.quantize_inference(output)

    def get_flops_and_params(self):
        
        if hasattr(self, "qencoder"):
            flops_e, params_e = self.qencoder.get_flops_and_params()
        if hasattr(self, "qdecoder"):
            flops_d, params_d = self.qdecoder.get_flops_and_params()

        total_flops = flops_e + flops_d
        total_params = params_e + params_d
        return total_flops, total_params
