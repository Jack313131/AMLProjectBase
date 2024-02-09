from torch.serialization import load
#from model import *

from PIL import Image, ImageOps
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_fx
from torch.autograd import Variable
from torchvision import datasets, transforms
from dataset import VOC12, cityscapes
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard
import os
import os.path as osp
import importlib
from shutil import copyfile
import copy
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20  # pascal=22, cityscapes=20

def direct_quantize(args, model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        output = model.quantize_forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')


# def full_inference(args, model, test_loader):
#     intersection = 0
#     union = 0
#     iouEvalVal = iouEval(NUM_CLASSES)
#     for i, (data, target) in enumerate(test_loader, 1):
#         if args.cuda:
#             data = data.cuda()
#             target = target.cuda()
#         data = Variable(data)
#         with torch.no_grad():
#             output = model(data)
#         # finalOutput = output.max(1)[1].unsqueeze(1)
#         # iouEvalVal.addBatch(finalOutput.data, target)
#         pred = output.argmax(dim=1)
#         intersection += torch.logical_and(pred, target).sum().item()
#         union += torch.logical_or(pred, target).sum().item()
#     miou = intersection / union
#     iouVal = 0
#     iouEvalVal = iouEval(NUM_CLASSES)
#     iouVal, iou_classes = iouEvalVal.getIoU()
#     iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
#     print("EPOCH IoU on VAL set: ", iouStr, "%")

#         # remember best valIoU and save checkpoint
#     if iouVal == 0:
#         current_acc = 0
#     else:
#         current_acc = iouVal

#     print(f"Current Acc : {current_acc}")
#     print('\nTest set: mIoU: {:.2f}%\n'.format(miou * 100))
#     return miou

def full_inference(args, model, test_loader):
    intersection = 0
    union = 0
    iouEvalVal = iouEval(NUM_CLASSES)
    for i, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        with torch.no_grad():
            output = model(data)
        # finalOutput = output.max(1)[1].unsqueeze(1)
        # iouEvalVal.addBatch(finalOutput.data, target)
            pred = output.argmax(dim=1)
            intersection += torch.logical_and(pred, target).sum().item()
            union += torch.logical_or(pred, target).sum().item()
    miou = intersection / union
    print('\nTest set: mIoU: {:.2f}%\n'.format(miou * 100))
    return miou

def quantize_inference(args, model, test_loader):
    correct = 0
    intersection = 0
    union = 0
    for i, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        with torch.no_grad():
            output = model.quantize_inference(data)
            pred = output.argmax(dim=1)
            intersection += torch.logical_and(pred, target).sum().item()
            union += torch.logical_or(pred, target).sum().item()
    miou = intersection / union
    print('\nTest set: mIoU: {:.2f}%\n'.format(miou * 100))
    return miou
    #print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

# def quantize_inference(args, model, test_loader):
#     correct = 0
#     for i, (data, target) in enumerate(test_loader, 1):
#         if args.cuda:
#             data = data.cuda()
#             target = target.cuda()
#         output = model.quantize_inference(data)
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#     #print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc  # A flag (True/False) to enable additional processing on the target image.
        self.augment = augment  # A flag to enable or disable augmentation.
        self.height = height  # The desired height to resize images.
        pass

    def __call__(self, input, target):  # method is executed when an instance of the class MyCoTransform is invoked

        input = Resize(self.height, Image.BILINEAR)(
            input)  # L'interpolazione bilineare (BILINEAR) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione bilineare considera i 4 pixel più vicini nella posizione corrispondente dell'immagine originale. Il valore del nuovo pixel è calcolato come una media ponderata dei valori di questi quattro pixel. Le ponderazioni sono basate sulla distanza relativa del punto calcolato rispetto a ciascuno di questi quattro pixel. In termini semplici, più un pixel è vicino al punto calcolato, maggiore sarà il suo contributo al valore finale.
        target = Resize(self.height, Image.NEAREST)(
            target)  # L'interpolazione nearest neighbor (Nearest) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione nearest neighbor semplicemente seleziona il valore del pixel più vicino nell'immagine originale, senza considerare altri pixel vicini. In altre parole, il valore del nuovo pixel è uguale a quello del pixel più vicino nella posizione corrispondente dell'immagine originale.

        input = ToTensor()(input)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = normalize(input)
        if (self.enc):
            target = Resize(int(self.height / 8), Image.NEAREST)(target)  # avviene un resize probabilmente per portare l'immagine ad avere dimensioni che poi saranno usate per la fase di convoluzione
        target = ToLabel()(target)
        # l'operazione di Relabel consiste per l'output target ovvero quello che già ha una classificazione per ogni pixel, di cambiare tutti i pixel con valore 255 a valore 19
        # questo perchè magari pixel 255 non ha una label associata mentre 19 è la label per classificazione generica (es : background)
        target = Relabel(255, 19)(target)

        return input, target
    
def main(args):
    enc = False
    drivedir = f'/content/drive/MyDrive/'
    save_file = drivedir + '[PTQ]/' + 'quantized_wights.pth'
    if not os.path.exists(drivedir):
        assert("Drivedir does not exist")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    # co_transform = MyCoTransform(enc, height=args.height)  # 1024)
    co_transform_val = MyCoTransform(enc, height=args.height)  # 1024)
    # dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    copyfile(args.model + ".py", drivedir + '/' + args.model + ".py")

    # train_loader = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    # )

    test_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        # Questa linea estrae lo stato attuale del modello, cioè i parametri attuali (pesi, bias, ecc.) del modello. state_dict() è una funzione PyTorch che restituisce un ordinato dizionario (OrderedDict) dei parametri.
        own_state = model.state_dict()
        # Il codice itera attraverso ogni coppia chiave-valore nel dizionario state_dict. name è il nome del parametro (per esempio, il nome di un particolare strato o peso nella rete), e param sono i valori effettivi dei pesi per quel nome.
        for name, param in state_dict.items():
            if name not in own_state:
                # Se il nome inizia con "module.", questo suggerisce che il dizionario dei pesi proviene da un modello che è stato addestrato usando DataParallel,
                # che aggiunge il prefisso "module." a tutti i nomi dei parametri. In questo caso, il codice cerca di adattare i nomi dei parametri rimuovendo "module." e tenta nuovamente di caricare il peso nel modello.
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(
                        param)  # name.split("module.")[-1] --> toglie la parte module. dal nome e si tiene il resto
                else:
                    print(name, " not loaded")
                    continue
            else:
                # se il modello partenza ha già un parametro con quel nome lo aggiorna direttamente
                # copy_ è una funzione in-place di PyTorch, il che significa che modifica direttamente il contenuto del tensore a cui si applica.
                # Poiché own_state[name] è un riferimento ai parametri reali del modello, questa operazione aggiorna direttamente i pesi all'interno del modello
                own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")


    model.eval()
    print("### Full inference ###")
    full_inference(args, model, test_loader)

    num_bits = 8
    print("### Model quantize ###")
    model.module.quantize(num_bits=num_bits)
    model.eval()
    print('Quantization bit: %d' % num_bits)


    # if load_quant_model_file is not None:
    #     model.load_state_dict(torch.load(load_quant_model_file))
    #     print("Successfully load quantized model %s" % load_quant_model_file)
    
    print("### Direct quantize ###")
    direct_quantize(args, model.module, test_loader)

    torch.save(model.state_dict(), save_file)
    print("### Model Saved ###")
    model.module.freeze()

    # 测试是否设备转移是否正确
    # model.cuda()
    # print(model.qconv1.M.device)
    # model.cpu()
    # print(model.qconv1.M.device)

    quantize_inference(model.module, test_loader)

    ################# Provato ad utilizzare il metodo quantize_fx => da problemi nella funzione fx.symbolic_trace
    # m = copy.deepcopy(model)
    # print("### Model copied ###")
    # m.eval()
    # qconfig_dict = {"": torch.quantization.get_default_qconfig(backend="fbgemm")}

    # example_input = (torch.rand(6, 3, args.height, args.height), torch.rand(6, 3, args.height, args.height))
    # model_prepared = m.module.prepare_fx(m, qconfig_dict, example_input)
    # model_prepared = torch.fx.symbolic_trace(model_prepared)
    # print("### Starting inference ###")
    # with torch.inference_mode():
    #     for i, (data, target) in enumerate(test_loader):
    #       if args.cuda:
    #           data = data.cuda()
    #           target = target.cuda()
    #       model_prepared(data)
    # #print(model_prepared.shape)
    # model_quantized = quantize_fx.convert_fx(model_prepared)
    # torch.save(model_quantized.state_dict(), save_file)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=True) 
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int,
                        default=50)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder')  # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--resume', action='store_true')  # Use this flag to load last checkpoint for training
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--cpu', action='store_true')
    main(parser.parse_args())
    