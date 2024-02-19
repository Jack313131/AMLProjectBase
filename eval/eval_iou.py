import torch
import os
import time
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, ToPILImage
from Loss import CrossEntropyLoss2d,weight
import utils as myutils
from dataset import cityscapes
from erfnet import Net
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry
from BiSeNetV1 import BiSeNetV1

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  # ignore label to 19
])

def main(args):

    print(f"Evaluation of mIoU using the metrics : {args.method}")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    if args.compareModel:
        assert args.loadModelPruned is not None , "Path for the mod model to compare is missing ..."
        path_model_mod = args.loadDir + args.loadModelPruned
        modelMod = Net(NUM_CLASSES)

    print("Loading model Original: " + modelpath)
    print("Loading weights Original: " + weightspath)

    if "bisenet" in args.loadModel.casefold().replace(" ", ""):
        modelOriginal = BiSeNetV1(NUM_CLASSES, 'train')
    if "erfnet" in args.model.casefold().replace(" ", ""):
        modelOriginal = Net(NUM_CLASSES)

    if (not args.cpu):
        modelOriginal = torch.nn.DataParallel(modelOriginal).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load modelOriginal when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    modelOriginal = load_my_state_dict(modelOriginal, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights Original LOADED successfully")

    if args.compareModel:
        print("Loading model and weights Mod: " + path_model_mod)
        modelMod, toFineTuningAdaptingLayer = myutils.remove_mask_from_model_with_pruning(modelMod, torch.load(path_model_mod,map_location=lambda storage, loc: storage))
        print("Model and weights Mod LOADED successfully")
        layer_names = list(modelMod.state_dict().keys())
        if any('adaptingInput' in name for name in layer_names) and toFineTuningAdaptingLayer==True:
            myutils.training_new_layer_adapting(model=modelMod,input_transform_cityscapes=input_transform_cityscapes,
                                            target_transform_cityscapes = target_transform_cityscapes, weight=weight,args=args)

        if args.typeQuantization == "int8":
            print("Applying quantization to int8 ... ")
            modelMod.eval()
            modelMod.set_config('x86')
            modelMod.quantize()
            print("Model Mod Quantized ...")
            calibration_dataloader = DataLoader(
                cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset='val'),
                num_workers=torch.cuda.device_count(), batch_size=6, shuffle=False)

            myutils.direct_quantize(args, modelMod, calibration_dataloader)
            modelMod.freeze()
            print("Model Mod calibrated ...")

        if args.typeQuantization == "float16":
            print("Applying model to float16 ... ")
            modelMod = modelMod.half()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalValOriginal = iouEval(NUM_CLASSES)
    if args.compareModel:
        iouEvalValMod = iouEval(NUM_CLASSES)

    if torch.cuda.is_available():
        modelOriginal = modelOriginal.to('cuda')
        if args.compareModel:
            modelMod = modelMod.to('cuda')


    modelOriginal.eval()
    if args.compareModel:
        modelMod.eval()

    start = time.time()
    print("\n\nStarting evaluation ....")
    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputsOriginal = modelOriginal(inputs)
            if "bisenet" in args.loadModel.casefold().replace(" ", ""):
                outputsOriginal = outputsOriginal[0]
            if args.compareModel:
                outputsMod = modelMod(inputs)
                finalOutputMod = outputsMod.max(1)[1].unsqueeze(1)
                iouEvalValMod.addBatch(finalOutputMod.data, labels)


        finalOutputOriginal = outputsOriginal.max(1)[1].unsqueeze(1)

        if args.method.casefold().replace(" ", "") == "msp":
            temperature = args.temperature
            scaledresult = outputsOriginal / temperature
            probs = torch.nn.functional.softmax(scaledresult, 1)  # result = modelOriginal(images), F = torch.nn.functional
            _, predicted_classes = torch.max(probs, dim=1)
            finalOutputOriginal = predicted_classes.unsqueeze(1)
        if args.method.casefold().replace(" ", "") == "maxentropy":
            eps = 1e-10
            probs = torch.nn.functional.softmax(outputsOriginal, dim=1)
            entropy = torch.div(torch.sum(-probs * torch.log(probs + eps), dim=1),
                                torch.log(torch.tensor(probs.shape[1]) + eps))
            confidence = 1 - entropy
            weighted_output = probs * confidence.unsqueeze(1)
            _, predicted_classes = torch.max(weighted_output, dim=1)
            finalOutputOriginal = predicted_classes.unsqueeze(1)

        iouEvalValOriginal.addBatch(finalOutputOriginal.data, labels)


        filenameSave = filename[0].split("leftImg8bit/")[1]


    iouValOriginal, iou_classes_original = iouEvalValOriginal.getIoU()
    if args.compareModel:
        iouValMod, iou_classes_mod = iouEvalValMod.getIoU()

    iou_classes_str_original = []
    for i in range(iou_classes_original.size(0)):
        iouStr = getColorEntry(iou_classes_original[i]) + '{:0.2f}'.format(iou_classes_original[i] * 100) + '\033[0m'
        iou_classes_str_original.append(iouStr)

    if args.compareModel:
        iou_classes_str_mod = []
        for i in range(iou_classes_mod.size(0)):
            iouStr = getColorEntry(iou_classes_mod[i]) + '{:0.2f}'.format(iou_classes_mod[i] * 100) + '\033[0m'
            iou_classes_str_mod.append(iouStr)

    text_model = ""
    path_result = f'../save/{args.loadModel}'
    class_name = ["Road", "sidewalk","building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    if args.compareModel:
        name_modules = " ".join(str(name_module) for name_module in args.listLayerPruning)
        num_layers = " ".join(str(num_layer) for num_layer in args.listNumLayerPruning)
        text_model = (f"The model is with pruning {args.typePruning} (amount : {args.pruning} & norm = {args.typeNorm}) "
                      f"for the modules :  {name_modules} applied on layers :  {num_layers}")
        dir_model = args.modelFilenameDrive.replace(".pth", "/")
        if myutils.is_drive_connect() == True:
            path_result = f"{args.path_drive}ModelsExtra/{args.load_dir_model_mod}/{dir_model}"

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    dir_save_result = f"{path_result}/results_miou.txt"
    print(f"Saving result on path : {dir_save_result}")

    with open(dir_save_result, 'w') as file:
        if text_model != "":
          myutils.print_and_save(text_model,file)
        myutils.print_and_save("---------------------------------------", file)
        myutils.print_and_save(f"Took {time.time() - start} seconds", file)
        myutils.print_and_save("=======================================", file)

        myutils.print_and_save("Per-Class IoU:", file)
        for i in range(len(iou_classes_str_original)):
            if args.compareModel:
                myutils.print_and_save(
                f"{iou_classes_str_original[i]} (ModelOriginal) - {iou_classes_str_mod[i]} (Model Pruned) -- {class_name[i]}",
                file)
            else:
                myutils.print_and_save(
                    f"{iou_classes_str_original[i]} -- {class_name[i]}",file)

        myutils.print_and_save("=======================================", file)
        iouStr = getColorEntry(iouValOriginal) + '{:0.2f}'.format(iouValOriginal * 100) + '\033[0m'
        if args.compareModel:
            iouModStr = getColorEntry(iouValMod) + '{:0.2f}'.format(iouValMod * 100) + '\033[0m'
            myutils.print_and_save(f"MEAN IoU: {iouStr}% (Model Original) --- MEAN IoU: {iouModStr}% (Model Pruned)", file)
            flopsOriginal, flopsPruning, paramsOriginal, paramsPrunning = myutils.compute_difference_flop(modelOriginal=modelOriginal,modelPruning=modelMod)
            myutils.print_and_save( f"\nFLOPs modelOriginal : {flopsOriginal} - FLOPs modelPruning : {flopsPruning} the difference is : {flopsOriginal - flopsPruning}",file)
            myutils.print_and_save(f"Params modelOriginal : {paramsOriginal} - Params modelPruning : {paramsPrunning} the difference is : {paramsOriginal - paramsPrunning}\n",file)
        else:
            myutils.print_and_save(f"MEAN IoU: {iouStr}%", file)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument("--loadModelPruned",default='')
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=not torch.cuda.is_available())
    parser.add_argument('--method', default='MaxLogit')
    parser.add_argument("--temperature", default=1.0)
    parser.add_argument("--pruning", type=float, default=0)
    parser.add_argument("--typeQuantization", type=str, default="float32")
    parser.add_argument("--compareModel", action='store_true')

    args = myutils.set_args(parser.parse_args())
    #myutils.connect_to_drive()

    path_project = "./"
    if os.path.exists('/content/AMLProjectBase'):
        path_project = '/content/AMLProjectBase/'
    if os.path.basename(os.getcwd()) != "eval":
        os.chdir(f"{path_project}eval")

    main(args)