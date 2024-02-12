import contextlib
import copy
import sys

import torch.nn as nn
import thop
import torch
import torch.nn.utils.prune as prune
import logging
import os
import re
from train.erfnet import non_bottleneck_1d,DownsamplerBlock


def convert_model_from_dataparallel(model):

    if isinstance(model, nn.DataParallel):
        model = model.module

    return model

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
def compute_difference_flop(modelOriginal,modelPruning):

    logger = logging.getLogger('thop')
    logger.setLevel(logging.WARNING)

    input = torch.randn(1, 3, 512, 1024).to("cuda" if torch.cuda.is_available() else "cpu")
    with suppress_stdout():
        flopsOriginal, paramsOriginal = thop.profile(modelOriginal, inputs=(input,))
        flopsPruning, paramsPrunning = thop.profile(modelPruning, inputs=(input,))
    print(f"FLOPs modelOriginal : {flopsOriginal} - FLOPs modelPruning : {flopsPruning} the difference is : {flopsOriginal-flopsPruning}")
    print(f"Params modelOriginal : {paramsOriginal} - Params modelPruning : {paramsPrunning} the difference is : {paramsOriginal - paramsPrunning}\n")

    return flopsOriginal,flopsPruning,paramsOriginal,paramsPrunning

def remove_mask_from_model_with_pruning(model,args):
    if "encoder" in args.moduleErfnetPruning:
        if args.typePruning.casefold().replace(" ", "") == "unstructured":
            if isinstance(model, torch.nn.DataParallel):
                for name, module in model.module.encoder.named_modules():
                    if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or (match and str(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bais')
                    if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                        if len(args.listNumLayerPruning) == 0 or (match and str(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bais')
            else:
                for name, module in model.encoder.named_modules():
                    if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or ( match and str(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bais')
                    if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                        if len(args.listNumLayerPruning) == 0 or ( match and str(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bais')
        elif args.typePruning.casefold().replace(" ", "") == "structured":
            if isinstance(model, torch.nn.DataParallel):
                for name, module in model.module.named_modules():
                    if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or ( match and int(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bais')
                    if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or (match and int(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(layer, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(layer, name='bias')
            else:
                for name, module in model.named_modules():
                    if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or (match and int(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(module, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(module, name='bias')
                    if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                        match = re.search(r'\d+', name)
                        if len(args.listNumLayerPruning) == 0 or ( match and int(match.group()) in args.listNumLayerPruning):
                            for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules() if any(substring in name2 for substring in args.listInnerLayerPruning)]:
                                prune.remove(module, name='weight')
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    prune.remove(module, name='bias')

    return model

def remove_prunned_channels_from_model(modelOriginal, args):
    modelOriginal = convert_model_from_dataparallel(modelOriginal)
    modelFinal = copy.deepcopy(modelOriginal)
    new_input_channel_next_layer = 0
    parent_module = None
    for nameLayer, layer in  modelOriginal.named_modules():
        if isinstance(layer, nn.Conv2d) and not 'output_conv' in nameLayer:
            if new_input_channel_next_layer > 0:
                in_channels = new_input_channel_next_layer
                input_changed = True
            else:
                in_channels = layer.in_channels
                input_changed = False
                non_zero_filters_prev_layer = None

            # Trova i filtri non azzerati (esempio ipotetico, la logica esatta può variare)
            non_zero_filters = layer.weight.data.sum(dim=(1, 2, 3)) != 0
            new_out_channels = non_zero_filters.long().sum().item()
            new_input_channel_next_layer = new_out_channels if new_out_channels != layer.out_channels else 0
            if new_out_channels != layer.out_channels or in_channels != layer.in_channels:

                if input_changed == False and layer.in_channels != new_out_channels:
                    new_layer_adapt_input = nn.Conv2d(layer.in_channels, new_out_channels, kernel_size=1, stride=1)
                    path_keys = str(nameLayer).split(".")
                    parent_module = modelFinal
                    for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                        parent_module = getattr(parent_module, key)

                    setattr(parent_module, "adaptingInput", new_layer_adapt_input)

                new_layer =  nn.Conv2d(in_channels=in_channels, out_channels=new_out_channels,
                      kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                      dilation=layer.dilation,bias=layer.bias is not None)

                if input_changed:
                    new_layer.weight.data = layer.weight.data[non_zero_filters][:,non_zero_filters_prev_layer,:,:]
                    if layer.bias is not None:
                        new_layer.bias.data = layer.bias.data[non_zero_filters]
                else:
                    new_layer.weight.data = layer.weight.data[non_zero_filters]
                    if layer.bias is not None:
                        new_layer.bias.data = layer.bias.data[non_zero_filters]


                path_keys = str(nameLayer).split(".")
                parent_module = modelFinal
                for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                    parent_module = getattr(parent_module, key)

                setattr(parent_module,path_keys[-1],new_layer)
                non_zero_filters_prev_layer = non_zero_filters

        if parent_module is not None and  isinstance(parent_module,DownsamplerBlock) and isinstance(layer, nn.Conv2d) and new_input_channel_next_layer == 0 :
            #new_input_channel_next_layer = new_layer.in_channels+new_layer.out_channels
            new_layer_adapt_input = nn.Conv2d(in_channels=new_layer.in_channels, out_channels=new_layer.out_channels, kernel_size=1, stride=1)
            setattr(parent_module, "adaptingInput", new_layer_adapt_input)

        if isinstance(layer, nn.BatchNorm2d) and  new_input_channel_next_layer > 0:
            new_bn = nn.BatchNorm2d(new_input_channel_next_layer, eps=layer.eps)
            path_keys = str(nameLayer).split(".")
            parent_module = modelFinal
            for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                parent_module = getattr(parent_module, key)
            setattr(parent_module,path_keys[-1],new_bn)

        if isinstance(layer,nn.ConvTranspose2d) and new_input_channel_next_layer >0 :
            new_layer = nn.ConvTranspose2d(in_channels=new_input_channel_next_layer, out_channels=layer.out_channels,
                                  kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                  output_padding = layer.output_padding,
                                  dilation=layer.dilation, bias=layer.bias is not None)
            path_keys = str(nameLayer).split(".")
            parent_module = modelFinal
            for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                parent_module = getattr(parent_module, key)
            setattr(parent_module, path_keys[-1], new_layer)
            new_input_channel_next_layer = 0


    return modelFinal
