# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from compute_flops import compute_flops

from module import *

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # print("In: {}, out: {}", ninput, noutput)
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.deq = torch.quantization.DeQuantStub()

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
        output = self.relu(output)
        #print("Relu: ", output.shape)
        #print("Output: ", output.shape)
        #print("#### END ####")
        return output

    def quantize(self, num_bits=8):
        self.block = torch.ao.quantization.fuse_modules(self, [['bn', 'relu']])
        self.block = torch.ao.quantization.prepare(self.block)
        return
        # self.qconv = QConv2d(self.conv, qi=True, qo=True, num_bits=num_bits)
        # self.qpool = QMaxPooling2d(self.pool.kernel_size, stride=self.pool.stride, padding=self.pool.padding, qi=True, num_bits=num_bits)
        self.qconvpool = QConvMaxP(self.conv, self.pool, qi=True, qo=True, num_bits=num_bits)
        self.qbn = QBN(self.bn, qi=False, qo=True)
        self.qrelu = QReLU()
        # 
        ############################
        # self.qconv = QConvMaxPoolBN2d(self.conv, self.pool, self.bn, qi=True, qo=True, num_bits=num_bits)
        # self.qrelu = QReLU()
        #oppure
        # self.qconv = QConvMaxPoolBNReLU(self.conv, self.pool, self.bn, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, input): ##calibrazione
        return self.block(input)
        # print("#### QDS ####")
        #print("Input: ", input.shape)
        # x = self.qconv(input)
        # print("QPool_f: ", x.shape)
        # x = self.qpool(input)
        # x = torch.cat([self.qconv(input), self.qpool(input)], 1)
        x = self.qconvpool(input)
        x = self.qbn(x)
        x = self.qrelu(x)
        #print("Qrelu_f: ", x.shape)
        #print("Output: ", x.shape)
        # print(x.shape)
        # x = self.qrelu(x)
        #print("#### END ####")
        return x

    def freeze(self): ##salvo qi e qo
        self.block = torch.ao.quantization.convert(self.block)
        return
        # print("##### Freezing DS #####")
        # self.qconv.freeze()
        # self.qpool.freeze()
        self.qconvpool.freeze()
        self.qbn.freeze(qi=self.qconvpool.qo)
        self.qrelu.freeze(self.qbn.qo)
        # self.qrelu.freeze(self.qconv.qo)

    def quantize_inference(self, x):
        return self.block(x)
        # print("#### QI DS ####")
        # print("Input: ", x.shape)
        qx = self.qconvpool.qi.quantize_tensor(x)
        # qx = self.qconv.quantize_inference(qx)
        # out = self.qconv.qo.dequanitze_tensor(qx)
        # qx_p = self.qpool.qi.quantize_tensor(x)
        # # print("Cat: ", qx.shape)
        # # print("Cat p: ", qx_p.shape)
        qx = self.qconvpool.quantize_inference(qx)
        # # print("Cat: ", qx.shape)
        # qx = self.qrelu.quantize_inference(qx)
        qx = self.qbn.quantize_inference(qx)
        qx = self.qrelu.quantize_inference(qx)
        out = self.qrelu.qi.dequantize_tensor(qx)
        # out = self.deq(qx)
        # out = self.qrelu.qo.dequantize_tensor(qx)
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
        self.quant = torch.quantization.QuantStub()
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
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, input):
        #print("#### NB1D ####")
        #print("Input: ", input.shape)
        output = self.conv3x1_1(input)
        # print("Conv: ", output.shape)
        output = self.relu1(output)
        # print("Relu: ", output.shape)

        output = self.conv1x3_1(output)
        # print("Conv: ", output.shape)
        output = self.bn1(output)
        # print("BN: ", output.shape)
        output = self.relu2(output)
        # print("Relu: ", output.shape)

        output = self.conv3x1_2(output)
        # print("Conv: ", output.shape)
        output = self.relu3(output)
        # print("Relu: ", output.shape)
        output = self.conv1x3_2(output)
        # print("Conv: ", output.shape)
        output = self.bn2(output)
        # print("BN: ", output.shape)

        if (self.dropout.p != 0):
            output = self.dropout(output)
            #print("Drop: ", output.shape)

        output = self.relu4(output+input)    #+input = identity (residual connection)
        # print("Relu: ", output.shape)
        #print("Output: ", output.shape)
        #print("#### END ####")
        return output
        #return F.relu(output+input)    #+input = identity (residual connection)

    def quantize(self, num_bits=8):
        self.block = torch.ao.quantization.fuse_modules(self, [
            ['conv3x1_1', 'relu1'],
            ['conv1x3_1', 'bn1', 'relu2'],
            ['conv3x1_2', 'relu3']])
        self.block = torch.ao.quantization.prepare(self.block)
        return
        self.qconv3x1_1 = QConv2d(self.conv3x1_1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qconv1x3_1 = QConvBNReLU(self.conv1x3_1, self.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv3x1_2 = QConv2d(self.conv3x1_2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qconv1x3_2 = QConv2d(self.conv1x3_2, qi=False, qo=True, num_bits=num_bits)
        self.qbn2 = QBN(self.bn2, qi=False, qo=True)
        self.qrelu3 = QReLU()
        # if not self.dropout.p == 0:
        #     self.qdropout = self.dropout
        # self.qconv3x1_2 = QConvBNReLU(self.conv3x1_2, self.bn2, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, input):
        return self.block(input)
       
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
        output = self.qbn2(output)

        # if hasattr(self, 'qdropout2d'):
        #     output = self.qdropout(output)
            #print("QDropout_f: ", output.shape)

        # Apply the identity (residual connection) and ReLU
        output = self.qrelu3(output + input)  # +x = identity (residual connection)
        # print("QRelu3_f: {}", output.shape)
        #print("Outpuy: ", output.shape)
        #print("#### END ####")
        return output

    def freeze(self):
        self.block = torch.ao.quantization.convert(self.block)
        return
        # print("##### Freezing NB1D #####")
        self.qconv3x1_1.freeze()
        self.qrelu1.freeze(self.qconv3x1_1.qo)
        self.qconv1x3_1.freeze(qi=self.qconv3x1_1.qo)
        self.qconv3x1_2.freeze(qi=self.qconv1x3_1.qo)
        self.qrelu2.freeze(self.qconv3x1_2.qo)
        self.qconv1x3_2.freeze(qi=self.qconv3x1_2.qo)
        # if hasattr(self, 'qdropout2d'):
        #     self.qdropout2d.freeze(self.qconv1x3_2.qo)
        self.qbn2.freeze(qi=self.qconv1x3_2.qo)
        self.qrelu3.freeze(self.qbn2.qo)
        # self.qrelu3.freeze(self.qconv3x1_2.qo)
        # self.qbn2.freeze()

    def quantize_inference(self, x):
        return self.block(x)
        # print("#### QI NB1D ####")
        # print("Input: ", x.shape)
        qx = self.qconv3x1_1.qi.quantize_tensor(x)
        # print("Con: ", qx.shape)
        qx = self.qconv3x1_1.quantize_inference(x)
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
        qx = self.qbn2.quantize_inference(qx)
        if hasattr(self, 'qdropout2d'):
            qx = self.quantize_inference(qx)
        # print("Conv: ", qx.shape)a
        qx_relu = self.qrelu3.qi.quantize_tensor(x)
        x_relu = self.qrelu3.qi.dequantize_tensor(qx)
        total_1 = qx+qx_relu
        total_2 = self.qrelu3.qi.quantize_tensor(x_relu+x)
        print(f"Total1: {total_1[0, 0, 0, :10]}\n Total2: {total_2[0, 0, 0, :10]}")
        qx = self.qrelu3.quantize_inference(total_1)
        out = self.qrelu3.qi.dequantize_tensor(qx)
        # out = self.qrelu3.qi.dequantize_tensor(qx)
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

    def forward(self, input, predict=False, file=None):
        if file:
            file.write("Input of Encoder (forward):\n")
            file.write(str(input[0, 0, 0, :10]) + "\n")  # Writing input value

        output = self.initial_block(input)
        if file:
            file.write(str(output[0, 0, 0, :10]) + "\n")  # Writing input value


        for layer in self.layers:
            output = layer(output)
            if file:
                file.write(str(output[0, 0, 0, :10]) + "\n")  # Writing input value

        if predict:
            output = self.output_conv(output)
        if file:
            file.write("End encoder\n")  # Writing input value
        return output

    
    def set_config(self, confg):
        self.initial_block.qconfig = torch.ao.quantization.get_default_qconfig(confg)
        for layer in self.layers:
            layer.qconfig = torch.ao.quantization.get_default_qconfig(confg)
        self.output_conv.qconfig = torch.ao.quantization.get_default_qconfig(confg)

    def quantize(self, num_bits=8):
        self.initial_block.quantize(num_bits=num_bits)
        for layer in self.layers:
            layer.quantize(num_bits=num_bits)
        # self.qoutput_conv = QConv2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, x, file=None):
        if file:
            file.write("Input of Encoder (quantize forward):\n")
            file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value
        # print("### Encoder (predict {}) ###".format(predict))
        x = self.initial_block.quantize_forward(x)
        
        if file:
            file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value
        for layer in self.layers:
            x = layer.quantize_forward(x)
            if file:
                file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value

            # print("### Output Conv ###")
        # x = self.qoutput_conv(x)
        return x

    def freeze(self):
        # print("#### Freezing Encoder ####")
        self.initial_block.freeze()
        for layer in self.layers:
            layer.freeze()
        # self.qoutput_conv.freeze()

    def quantize_inference(self, x, file):
        # print("#### Quantize Inference Encoder ####")
        # qx = self.initial_block.qconv.qi.quantize_tensor(x)
        if file:
            file.write("Input of Encoder (quantize_inference): \n")  # Writing input value
            file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value
        qx = self.initial_block.quantize_inference(x)

        if file:
            file.write(str(qx[0, 0, 0, :10]) + "\n")  # Writing input value

        # Pass the quantized input through each layer
        for layer in self.layers:
            qx = layer.quantize_inference(qx)
            if file:
                file.write(str(qx[0, 0, 0, :10]) + "\n")  # Writing input value

        # Pass the output through the output_conv layer if it exists
        # qx = self.qoutput_conv.quantize_inference(qx)
        if file:
            file.write("End Encoder\n")  # Writing input value
        return qx
    
    def get_flops_and_params(self):
        flops, params = self.initial_block.get_flops_and_params()
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
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.conv.qconfig = torch.quantization.default_qconfig
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        # print("#### US ####")
        # print("Input: ", input.shape)
        output = self.conv(input)
        # print("Conv: ", output.shape)
        output = self.bn(output)
        # print("Bn: ", output.shape)
        output =  self.relu(output)
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
    
    def set_config(self, confg):
        for layer in self.layers:
            layer.qcongif = torch.ao.quantization.get_default_qconfig(confg)
        self.output_conv.qconfig = torch.ao.quantization.get_default_qconfig(confg)
    
    def forward(self, input, file):
        if file:
            file.write("Input decoder (forward):\n")  # Writing input value
            file.write(str(input[0, 0, 0, :10]) + "\n")  # Writing input value
        output = input

        for layer in self.layers:
            output = layer(output)
            if file:
                file.write(str(output[0, 0, 0, :10]) + "\n")  # Writing input value

        output = self.output_conv(output)
        if file:
            file.write(str(output[0, 0, 0, :10]) + "\n")  # Writing input value

        return output

    def quantize(self, num_bits=8):
        for layer in self.layers:
            if hasattr(layer, 'quantize'):
                layer.quantize(num_bits=num_bits)

        self.q_output_conv = QConvTranspose2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)
        
    
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

    def quantize_inference(self, x, file):
        # print("#### Quantize Inference Decoder ####")
        qx = x
        if file:
            file.write("Input decoder (quantize inference):\n")  # Writing input value
            file.write(str(qx[0, 0, 0, :10]) + "\n")  # Writing input value
        for layer in self.layers:
            if hasattr(layer, 'quantize_inference'):
                qx = layer.quantize_inference(qx)
                if file:
                    file.write(str(qx[0, 0, 0, :10]) + "\n")  # Writing input value
        qx = self.q_output_conv.qi.quantize_tensor(qx)
        qx = self.q_output_conv.quantize_inference(qx)
        out = self.q_output_conv.qo.dequantize_tensor(qx)
        if file:
            file.write(str(qx[0, 0, 0, :10]) + "\n")  # Writing input value
            file.write("End decoder\n")  # Writing input value

        return out
    
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
    def set_config(self, confg):
        self.encoder.set_config(confg)
        self.decoder.set_config(confg)

    def quantize(self, num_bits=8):
        self.encoder.quantize(num_bits=num_bits)
        self.decoder.quantize(num_bits=num_bits)
        
        # qencoder_params = list(self.encoder.parameters())
        # self.register_parameter('qencoder', nn.Parameter(qencoder_params))
        # self.register_parameter('qdecoder', nn.Parameter(self.encoder.parameters()))
            
    def forward(self, input, file, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder.forward(input, file=file)
            return self.decoder.forward(output, file=file)

    def quantize_forward(self, x):
        # print("Input: ", x.shape)
        output = self.encoder.quantize_forward(x)
        # print("Output: ", x.shape)
        return self.decoder.quantize_forward(output)
        # return self.decoder.quantize_forward(output)

    def freeze(self):
        self.encoder.freeze()
        self.decoder.freeze()

    def quantize_inference(self, x, file=None):
        # print("### Encoder input: ", x.shape)
        output = self.encoder.quantize_forward(x, file=file)
        # output = self.encoder.quantize_inference(x)
        # output = self.decoder.forward(output)
        output = self.decoder.quantize_inference(output, file=file)
        return output

    def get_flops_and_params(self):
        
        if hasattr(self, "qencoder"):
            flops_e, params_e = self.encoder.get_flops_and_params()
        if hasattr(self, "qdecoder"):
            flops_d, params_d = self.decoder.get_flops_and_params()

        total_flops = flops_e + flops_d
        total_params = params_e + params_d
        return total_flops, total_params
