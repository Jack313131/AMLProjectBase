import os
import random
import time
import numpy as np
import torch
import gc
import torch.nn.utils.prune as prune
from PIL import Image, ImageOps
from argparse import ArgumentParser
import re
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard
from erfnet import non_bottleneck_1d, DownsamplerBlock
import importlib
from iouEval import iouEval, getColorEntry
import utils as myutils
from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20  # pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


# Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc  # A flag (True/False) to enable additional processing on the target image.
        self.augment = augment  # A flag to enable or disable augmentation.
        self.height = height  # The desired height to resize images.
        pass

    def __call__(self, input, target):  # method is executed when an instance of the class MyCoTransform is invoked

        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if (self.augment):
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            transX = random.randint(-2,2)  # define randomly how much shift the images from 2 pixel to the left to 2 pixel to the right (could be also 0)
            transY = random.randint(-2,2)  # define randomly how much shift the images from 2 pixel to the bottom to 2 pixel to the up (could be also 0)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0),fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0),fill=255)

            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = normalize(input)
        if (self.enc):
            target = Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None, ignores_index=None):
        super().__init__()
        if ignores_index is not None:
            self.loss = torch.nn.NLLLoss(weight,ignore_index=ignores_index)
        else:
            self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):

        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)  # + reg_term


def get_class_weights(loader, num_classes, c=1.02):
    '''
    This class return the class weights for each class
    
    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration

    - num_classes : The number of classes

    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''

    _, (_, labels) = next(loader)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights_tensor

def train(args, model, enc=False):
    best_acc = 0
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726
        weight[1] = 4.4237880706787
        weight[2] = 2.9691488742828
        weight[3] = 5.3442072868347
        weight[4] = 5.2983593940735
        weight[5] = 5.2275490760803
        weight[6] = 5.4394111633301
        weight[7] = 5.3659925460815
        weight[8] = 3.4170460700989
        weight[9] = 5.2414722442627
        weight[10] = 4.7376127243042
        weight[11] = 5.2286224365234
        weight[12] = 5.455126285553
        weight[13] = 4.3019247055054
        weight[14] = 5.4264230728149
        weight[15] = 5.4331531524658
        weight[16] = 5.433765411377
        weight[17] = 5.4631009101868
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965
        weight[1] = 6.9850029945374
        weight[2] = 3.7890393733978
        weight[3] = 9.9428062438965
        weight[4] = 9.7702074050903
        weight[5] = 9.5110931396484
        weight[6] = 10.311357498169
        weight[7] = 10.026463508606
        weight[8] = 4.6323022842407
        weight[9] = 9.5608062744141
        weight[10] = 7.8698215484619
        weight[11] = 9.5168733596802
        weight[12] = 10.373730659485
        weight[13] = 6.6616044044495
        weight[14] = 10.260489463806
        weight[15] = 10.287888526917
        weight[16] = 10.289801597595
        weight[17] = 10.405355453491
        weight[18] = 10.138095855713

    weight[19] = 0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc, augment=True, height=args.height)  # 1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)  # 1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda and torch.cuda.is_available():
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    # TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    optimizer = Adam(model.parameters(), 5e-5, (0.9, 0.999), eps=1e-08, weight_decay=5e-5)  ## scheduler 1
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)

    start_epoch = 1
    if args.resume:
        # Must load weights, optimizer, epoch and best value.
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        if args.pruning > 0 and 'model' in checkpoint:
            model = checkpoint['model']
        elif "state_dict" in checkpoint and not 'model' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        if 'scheduler' in checkpoint:
          scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))
        if args.pruning > 0:
            print(f"Pruning Applied :")
            for name, module in model.module.named_modules():
                if args.typePruning == "unstructured":
                    total = module.weight.nelement()
                    zeros = torch.sum(module.weight == 0)
                    print(f"Name {name} Applied Pruning Unstructured Weight: {hasattr(module, 'weight_mask')} with value : {(zeros.float() / total) * 100:.2f}% di zero weights")
                if args.typePruning == "structured" and hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
                    original_out_channels = module.weight.size()[0]
                    non_zero_filters = module.weight_mask.data.sum(dim=(1, 2, 3)) != 0
                    new_out_channels = non_zero_filters.long().sum().item()
                    if new_out_channels != original_out_channels:
                        print(f"Name {name} Applied Pruning Structured with Norm {args.typeNorm} and pruning of {args.pruning * 100}% passing from {original_out_channels} to {new_out_channels} layers")
        del checkpoint
        gc.collect()

    if not args.resume and args.pruning > 0:

        print(f"Loading weigths from {args.loadWeights} ... ")

        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_( param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        model = load_my_state_dict(model, torch.load(args.loadWeights, map_location=lambda storage, loc: storage))
        print("Model and weights LOADED successfully")

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    pruning_setting_path = savedir + "/pruning_setting.txt"
    if args.pruning > 0 and not args.resume:
        if "encoder" in args.moduleErfnetPruning:
            if args.typePruning.casefold().replace(" ", "") == "unstructured":
                print(f"Applying pruning encoder (type pruning : {args.typePruning}) with value : {args.pruning} ... ")
                print(f"For more info of the prunning applied see the file : {pruning_setting_path}")
                with open(pruning_setting_path, 'w') as file:
                    file.write(
                        f"Applying pruning encoder (type pruning : {args.typePruning}) with value : {args.pruning} ... \n\n")
                if isinstance(model, torch.nn.DataParallel):
                    for name, module in model.module.encoder.named_modules():
                        if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    textFile = f"Module : non_bottleneck_1d , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    prune.l1_unstructured(layer, name='weight', amount=args.pruning)
                                    if hasattr(layer, 'bias') and layer.bias is not None:
                                        prune.l1_unstructured(module, name='bais', amount=args.pruning)
                                        textFile = textFile + " and bias"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                        if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    textFile = f"Module : DownsamplerBlock , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    prune.l1_unstructured(layer, name='weight', amount=args.pruning)
                                    if hasattr(layer, 'bias') and layer.bias is not None:
                                        prune.l1_unstructured(module, name='bais', amount=args.pruning)
                                        textFile = textFile + " and bias"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                else:
                    for name, module in model.encoder.named_modules():
                        if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    textFile = f"Module : non_bottleneck_1d , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    prune.l1_unstructured(layer, name='weight', amount=args.pruning)
                                    if hasattr(layer, 'bias') and layer.bias is not None:
                                        prune.l1_unstructured(layer, name='bias')
                                        textFile = textFile + " and bias"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                        if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    textFile = f"Module : DownsamplerBlock , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    prune.l1_unstructured(layer, name='weight', amount=args.pruning)
                                    if hasattr(layer, 'bias') and layer.bias is not None:
                                        prune.l1_unstructured(layer, name='bais', amount=args.pruning)
                                        textFile = textFile + " and bias"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
            elif args.typePruning.casefold().replace(" ", "") == "structured":
                print(f"Applying pruning encoder (type pruning : {args.typePruning})  with value : {args.pruning} ... ")
                print(f"For more info of the prunning applied see the file : {pruning_setting_path}")
                with open(pruning_setting_path, 'w') as file:
                    file.write(f"Applying pruning encoder (type pruning : {args.typePruning}) with value : {args.pruning} ... \n\n")
                if isinstance(model, torch.nn.DataParallel):
                    for name, module in model.module.encoder.named_modules():
                        if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    prune.ln_structured(layer, name='weight', amount=args.pruning, n=args.typeNorm,
                                                        dim=0)
                                    textFile = f"Module : non_bottleneck_1d , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                        if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    prune.ln_structured(layer, name='weight', amount=args.pruning, n=args.typeNorm,
                                                        dim=0)
                                    textFile = f"Module : DownsamplerBlock , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                else:
                    for name, module in model.encoder.named_modules():
                        if isinstance(module, non_bottleneck_1d) and "non_bottleneck_1d" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    prune.ln_structured(layer, name='weight', amount=args.pruning, n=args.typeNorm,
                                                        dim=0)
                                    textFile = f"Module : non_bottleneck_1d , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
                        if isinstance(module, DownsamplerBlock) and "DownsamplerBlock" in args.listLayerPruning:
                            match = re.search(r'\d+', name)
                            if len(args.listNumLayerPruning) == 0 or (
                                    match and str(match.group()) in args.listNumLayerPruning):
                                for nameLayer, layer in [(name2, module2) for name2, module2 in module.named_modules()
                                                         if any(
                                            substring in name2 for substring in args.listInnerLayerPruning)]:
                                    prune.ln_structured(layer, name='weight', amount=args.pruning, n=args.typeNorm,
                                                        dim=0)
                                    textFile = f"Module : DownsamplerBlock , Num_Layer : {name} , innerModule : {nameLayer} applying pruning on weight"
                                    with open(pruning_setting_path, 'a') as file:
                                        file.write(textFile + "\n")
            else:
                raise ValueError("No type of pruning specified between {unstructured-structured}")
        elif "decoder" in args.moduleErfnetPruning:
            print()
        else:
            raise ValueError("No module for pruning specified between {encoder,decoder}")

        with open(pruning_setting_path, 'a') as file:
            file.write(f"\n\nPruning Results : \n")
        for name, module in (
        model.module.named_modules() if isinstance(model, torch.nn.DataParallel) else model.named_modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
                if args.typePruning == "unstructured":
                    total = module.weight.nelement()
                    zeros = torch.sum(module.weight == 0)
                    with open(pruning_setting_path, 'a') as file:
                        file.write(
                            f"Name {name} Applied Pruning Unstructured Weight: {hasattr(module, 'weight_mask')} with value : {(zeros.float() / total) * 100:.2f}% di zero weights\n")
                if args.typePruning == "structured" and hasattr(module, 'weight') and hasattr(module,'weight_mask'):
                    original_out_channels = module.weight.size()[0]
                    non_zero_filters = module.weight_mask.data.sum(dim=(1, 2, 3)) != 0
                    new_out_channels = non_zero_filters.long().sum().item()
                    if new_out_channels != original_out_channels:
                        with open(pruning_setting_path, 'a') as file:
                            file.write(
                                f"Name {name} Applied Pruning Structured with Norm {args.typeNorm} and pruning of {args.pruning*100}% passing from {original_out_channels} to {new_out_channels} layers\n")


    for epoch in range(start_epoch, args.num_epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain
        doIouVal = args.iouVal

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            images.requires_grad_(True)
            inputs = images
            targets = labels

            if "BiSeNet" in args.model:
                outputs = model(inputs)
                outputs = outputs[0]
                outputs = outputs.float()
            if "erfnet" in args.model:
                outputs = model(inputs, only_encode=enc)
            if "ENet" in args.model:
                outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            scheduler.step()  ## scheduler 2

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)


            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):  # merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                            f'target (epoch: {epoch}, step: {step})')
                print("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain) + '{:0.2f}'.format(iouTrain * 100) + '\033[0m'
            print("EPOCH IoU on TRAIN set: ", iouStr, "%")

            # Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            with torch.no_grad():
                start_time = time.time()
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images.requires_grad_(True)
                inputs = images

                targets = labels

                if "BiSeNet" in args.model:
                    outputs = model(inputs)
                    outputs = outputs[0]
                    outputs = outputs.float()
                if "erfnet" in args.model:
                    outputs = model(inputs, only_encode=enc)
                if "ENet" in args.model:
                    outputs = model(inputs)

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)

                if (doIouVal):
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                    start_time_plot = time.time()
                    image = inputs[0].cpu().data
                    board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                    if isinstance(outputs, list):  # merge gpu tensors
                        board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                                    f'VAL output (epoch: {epoch}, step: {step})')
                    else:
                        board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                                    f'VAL output (epoch: {epoch}, step: {step})')
                    board.image(color_transform(targets[0].cpu().data),
                                f'VAL target (epoch: {epoch}, step: {step})')
                    print("Time to paint images: ", time.time() - start_time_plot)
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
            print("EPOCH IoU on VAL set: ", iouStr, "%")
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal

        print(f"Current Acc : {current_acc} - Best Acc : {best_acc}")
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'model': model,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best_{args.modelFilenameDrive}.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))

                    # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr))
        if args.saveCheckpointDriveAfterNumEpoch > 0 and step > 0 and step % args.saveCheckpointDriveAfterNumEpoch == 0:
            modelFilenameDrive = args.modelFilenameDrive
            myutils.saveOnDrive(epoch=epoch, model=modelFilenameDrive, args=args,
                        pathOriginal=f"/content/AMLProjectBase/save/{args.savedir}/")

    return (model)  # return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    if "BiSeNet" in args.model:
        model = model_file.BiSeNetV1(NUM_CLASSES, 'train')
    if "erfnet" in args.model:
        model = model_file.Net(NUM_CLASSES)
    if "ENet" in args.model:
        model = model_file.ENet(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        try:
            checkpoint = torch.load(args.state)
            state_dict = checkpoint['state_dict']
            # print(list(state_dict.keys()))  # Print the keys of the OrderedDict
            if 'fullconv.weight' in state_dict:
              print("IN")
              weight = state_dict['fullconv.weight']
              if weight.size() != (16, 20, 3, 3):
                print("Resize")
            # Resize the tensor to match the current model architecture
                resized_weight = weight.new_zeros((16, 20, 3, 3))
                resized_weight[:, :19, :, :] = weight  # Copy existing values
                state_dict['fullconv.weight'] = resized_weight
            print(state_dict['fullconv.weight'].shape)
            state_dict = {"module."+k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """

        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        # print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder and args.model.casefold().replace(" ", "") == "erfnet":
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()  # because loaded encoder is probably saved in cuda
        if not args.pretrainedEncoder and args.model.casefold().replace(" ", "") == "erfnet":
            pretrainedEnc = next(model.children()).encoder if isinstance(model,torch.nn.DataParallel) else next(model.children())
        if args.model.casefold().replace(" ", "") == "erfnetbarlowtwins":
            model = model_file.Net(NUM_CLASSES, encoder=None, batch_size=args.batch_size, backbone=args.backbone)
        if args.model.casefold().replace(" ", "") == "BiSeNet":
            model = model_file.BiSeNetV1(NUM_CLASSES, 'train')
        if args.model.casefold().replace(" ", "") == "erfnet":
            model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  # Add decoder to encoder
        if args.cuda and torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
        model = train(args, model, False)  # Train decoder
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=torch.cuda.is_available())  # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int,
                        default=50)  # variabile per determinare se e con quale frequenza visualizzare le metriche o le immagini durante l'addestramento (minore di 0 nessuna visualizzazione)
    parser.add_argument('--epochs-save', type=int, default=50)  # You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=False)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder')  # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize',
                        action='store_true')  # variabile per determinare se la visualizzazione è attivata o meno.

    parser.add_argument('--iouTrain', action='store_true',  # boolean to compute IoU evaluation  in the training phase
                        default=False)  # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true',
                        default=True)  # boolean to compute IoU evaluation also in the validation phase
    parser.add_argument('--resume', action='store_true')  # Use this flag to load last checkpoint for training
    parser.add_argument("--freezingBackbone", action='store_true')
    parser.add_argument("--saveCheckpointDriveAfterNumEpoch", type=int, default=1)
    parser.add_argument("--pruning", type=float, default=0.5)
    parser.add_argument("--typePruning", type=str, default="unstructured")
    parser.add_argument("--listInnerLayerPruning", nargs='+', default=['conv', 'bn'])
    parser.add_argument("--listLayerPruning", nargs='+', default=['non_bottleneck_1d', 'DownsamplerBlock'])
    parser.add_argument("--listWeight", nargs='+', default=['weight'])
    parser.add_argument("--typeNorm", type=int, default=None)
    parser.add_argument("--listNumLayerPruning", nargs='+', help='', default=[])
    parser.add_argument("--moduleErfnetPruning", nargs='+', help='Module List', default=[])
    parser.add_argument('--loadWeights', default="../trained_models/erfnet_pretrained.pth")

    path_project = "./"
    if os.path.exists('/content/AMLProjectBase'):
        path_project = '/content/AMLProjectBase/'
    if os.path.basename(os.getcwd()) != "train":
        os.chdir(f"{path_project}train")

    args = myutils.set_args(parser.parse_args())
    #myutils.connect_to_drive()

    main(args)