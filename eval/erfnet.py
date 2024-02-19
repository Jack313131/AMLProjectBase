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


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # print("In: {}, out: {}", ninput, noutput)
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.deq = torch.quantization.DeQuantStub()

    def forward(self, input):
        inputMaxPool = input
        if hasattr(self, "adaptingInput") and self.conv.in_channels + self.conv.out_channels != self.bn.num_features:
            inputMaxPool = self.adaptingInput(input)
        if hasattr(self, "maskInput") and self.conv.in_channels + self.conv.out_channels != self.bn.num_features:
            output_size_channels = self.conv(input).size(1)
            new_tensor = torch.zeros(input.size(0), output_size_channels, input.size(2), input.size(3))
            selected_indices = torch.where(self.maskInput)[0]
            if len(selected_indices) > input.size(1):
                raise ValueError("La maschera seleziona più canali di quanti ce ne siano nell'input")
            new_tensor[:, selected_indices, :, :] = input
            inputMaxPool = new_tensor

        output = torch.cat([self.conv(input), self.pool(inputMaxPool)], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output

    def quantize(self):
        self.block = torch.ao.quantization.fuse_modules(self, [['bn', 'relu']])
        self.block = torch.ao.quantization.prepare(self.block)
        return
        # self.qconvpool = QConvMaxP(self.conv, self.pool, qi=True, qo=True, num_bits=num_bits)
        # self.qbn = QBN(self.bn, qi=False, qo=True)
        # self.qrelu = QReLU()

    def quantize_forward(self, input):  ##calibrazione
        return self.block(input)
        # x = self.qconvpool(input)
        # x = self.qbn(x)
        # x = self.qrelu(x)
        # return x

    def freeze(self):  ##salvo qi e qo
        self.block = torch.ao.quantization.convert(self.block)
        return
        # self.qconvpool.freeze()
        # self.qbn.freeze(qi=self.qconvpool.qo)
        # self.qrelu.freeze(self.qbn.qo)

    def quantize_inference(self, x):
        return self.block(x)
        # qx = self.qconvpool.qi.quantize_tensor(x)
        # qx = self.qconvpool.quantize_inference(qx)
        # qx = self.qbn.quantize_inference(qx)
        # qx = self.qrelu.quantize_inference(qx)
        # out = self.qrelu.qi.dequantize_tensor(qx)
        # return out


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.relu1 = nn.ReLU()
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.relu2 = nn.ReLU()
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))
        self.relu3 = nn.ReLU()
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.relu4 = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

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

        if hasattr(self, 'maskInput') and input.size()[1] != output.size()[1]:
            if input.size()[1] > output.size()[1]:
                input = input[:,self.maskInput,:,:]
            elif input.size()[1] < output.size()[1]:
                new_tensor = torch.zeros(input.size()[0], output.size()[1], input.size()[2], input.size()[3])
                selected_indices = torch.where(self.maskInput)[0]
                if len(selected_indices) > input.size(1):
                    raise ValueError("La maschera seleziona più canali di quanti ce ne siano nell'input")
                new_tensor[:, selected_indices, :, :] = input
                input = new_tensor

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = self.relu4(output + input)  # +input = identity (residual connection)
        return output

    def quantize(self):
        self.block = torch.ao.quantization.fuse_modules(self, [
            ['conv3x1_1', 'relu1'],
            ['conv1x3_1', 'bn1', 'relu2'],
            ['conv3x1_2', 'relu3']])
        self.block = torch.ao.quantization.prepare(self.block)
        return
        # self.qconv3x1_1 = QConv2d(self.conv3x1_1, qi=True, qo=True, num_bits=num_bits)
        # self.qrelu1 = QReLU()
        # self.qconv1x3_1 = QConvBNReLU(self.conv1x3_1, self.bn1, qi=False, qo=True, num_bits=num_bits)
        # self.qconv3x1_2 = QConv2d(self.conv3x1_2, qi=False, qo=True, num_bits=num_bits)
        # self.qrelu2 = QReLU()
        # self.qconv1x3_2 = QConv2d(self.conv1x3_2, qi=False, qo=True, num_bits=num_bits)
        # self.qbn2 = QBN(self.bn2, qi=False, qo=True)
        # self.qrelu3 = QReLU()

    def quantize_forward(self, input):
        return self.block(input)
        # output = self.qconv3x1_1(input)
        # output = self.qrelu1(output)

        # output = self.qconv1x3_1(output)

        # output = self.qconv3x1_2(output)
        # output = self.qrelu2(output)

        # output = self.qconv1x3_2(output)
        # output = self.qbn2(output)
        # output = self.qrelu3(output + input)  # +x = identity (residual connection)
        # return output

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
        # qx = self.qconv3x1_1.qi.quantize_tensor(x)
        # qx = self.qconv3x1_1.quantize_inference(x)
        # qx = self.qrelu1.quantize_inference(qx)
        # qx = self.qconv1x3_1.quantize_inference(qx)
        # qx = self.qconv3x1_2.quantize_inference(qx)
        # qx = self.qrelu2.quantize_inference(qx)
        # qx = self.qconv1x3_2.quantize_inference(qx)
        # qx = self.qbn2.quantize_inference(qx)
        # if hasattr(self, 'qdropout2d'):
        #     qx = self.quantize_inference(qx)
        # qx_relu = self.qrelu3.qi.quantize_tensor(x)
        # qx = self.qrelu3.quantize_inference(qx+qx_relu)
        # out = self.qrelu3.qi.dequantize_tensor(qx)
        # return out


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

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

    def quantize(self):
        self.initial_block.quantize()
        for layer in self.layers:
            layer.quantize()
        # self.qoutput_conv = QConv2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)

    def quantize_forward(self, x, file=None):
        if file:
            file.write("Input of Encoder (quantize forward):\n")
            file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value
        x = self.initial_block.quantize_forward(x)

        if file:
            file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value
        for layer in self.layers:
            x = layer.quantize_forward(x)
            if file:
                file.write(str(x[0, 0, 0, :10]) + "\n")  # Writing input value

        return x

    def freeze(self):
        self.initial_block.freeze()
        for layer in self.layers:
            layer.freeze()

    def quantize_inference(self, x, file):
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

        # qx = self.qoutput_conv.quantize_inference(qx)
        if file:
            file.write("End Encoder\n")  # Writing input value
        return qx


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

    def quantize(self, num_bits=8):
        self.qconv = QConvTBNReLU(self.conv, self.bn, qi=True, qo=True, num_bits=num_bits)
        return self

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

    def quantize(self):
        for layer in self.layers:
            if hasattr(layer, 'quantize'):
                layer.quantize()

        # self.q_output_conv = QConvTranspose2d(self.output_conv, qi=True, qo=True, num_bits=num_bits)

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

    def quantize_inference(self, x, file):
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


# ERFNet
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

    def quantize(self):
        self.encoder.quantize()
        self.decoder.quantize()

        # qencoder_params = list(self.encoder.parameters())
        # self.register_parameter('qencoder', nn.Parameter(qencoder_params))
        # self.register_parameter('qdecoder', nn.Parameter(self.encoder.parameters()))

    def forward(self, input, file=None, only_encode=False):
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
