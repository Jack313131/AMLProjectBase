import contextlib
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
