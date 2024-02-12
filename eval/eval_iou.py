# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import utils.utils as myutils
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from train.erfnet import Net
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

# La classe ToPILImage è utilizzata per convertire tensori (tipicamente di PyTorch) o array ndarray (di NumPy) in immagini PIL. Questo è utile in molte applicazioni di elaborazione delle immagini dove si ha bisogno di convertire i formati dei dati tra diverse librerie.
# Gestione delle Dimensioni e dei Canali: La classe gestisce tensori con diverse configurazioni di canali (C x H x W) e array ndarray con configurazioni (H x W x C), dove C è il numero di canali, H è l'altezza e W è la larghezza dell'immagine.
# A seconda del numero di canali (C), la classe fa delle ipotesi sul modo (o sul formato del colore) dell'immagine risultante. Ad esempio, 4 canali suggeriscono un formato RGBA, mentre 3 canali suggeriscono RGB.
# Questa conversione è comune nel pre-processamento o post-processamento in applicazioni di visione artificiale, ad esempio, per visualizzare l'output di un modello di deep learning, salvare l'output in un formato di file immagine standard, o ulteriormente elaborare l'immagine usando librerie che lavorano con il formato PIL.
image_transform = ToPILImage()

# La classe Compose prende una lista di trasformazioni e le combina in una singola trasformazione. Quando viene chiamata su un'immagine, applica sequenzialmente ciascuna trasformazione della lista all'immagine. Le trasfomazioni nello specifico sono:
# 1 ) Resize(512, Image.BILINEAR) ---> Ridimensiona l'immagine a una dimensione specifica (in questo caso, 512 pixel per lato).
# 1a) Image.BILINEAR ---> L'interpolazione è un metodo utilizzato per ridimensionare le immagini. Quando si ingrandisce o si riduce un'immagine, è necessario calcolare i valori dei pixel nelle nuove posizioni. L'interpolazione determina questi valori basandosi sui pixel esistenti.
#                         Il termine "bilineare" si riferisce a un metodo di interpolazione che considera i 4 pixel più vicini alla posizione target (due in orizzontale e due in verticale) per calcolare il valore del nuovo pixel. Esegue i seguenti passaggi:
#                         Calcola i valori intermedi facendo una media lineare lungo una direzione (ad esempio, orizzontale). Ripete il processo nella direzione perpendicolare (ad esempio, verticale) utilizzando i valori intermedi precedentemente calcolati. Combina questi valori per ottenere il valore finale del pixel.
# 2 ) ToTensor() ---> Converte l'immagine (tipicamente un'immagine PIL o un array NumPy) in un tensore PyTorch. Durante questa conversione, i valori dei pixel vengono automaticamente scalati da un range [0, 255] (tipico delle immagini PIL) a un range [0, 1] (usato nei tensori PyTorch).
#                     Inoltre, riordina le dimensioni dell'immagine da HWC (Altezza, Larghezza, Canali) a CHW (Canali, Altezza, Larghezza), che è il formato standard per le immagini in PyTorch.
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

# La classe Compose prende una lista di trasformazioni e le combina in una singola trasformazione. Quando viene chiamata su un'immagine, applica sequenzialmente ciascuna trasformazione della lista all'immagine. Le trasfomazioni nello specifico sono:
# 1 ) Resize(512, Image.NEAREST) ---> Ridimensiona l'immagine a una dimensione specifica (in questo caso, 512 pixel per lato).
# 1a) Image.NEAREST --->  Quando un'immagine viene ridimensionata (ingrandita o rimpicciolita), è necessario calcolare i valori dei nuovi pixel. Il metodo "nearest neighbor" assegna al nuovo pixel il valore del pixel più vicino dal punto di vista geometrico nell'immagine originale.
#                         In pratica, significa che per ogni pixel nel nuovo spazio dell'immagine, il metodo cerca il pixel più vicino nel vecchio spazio dell'immagine e copia il suo valore.
#                         Vantaggi : Il metodo è molto semplice e veloce, il che lo rende utile per applicazioni in cui la velocità è più importante della qualità dell'immagine o quando si lavora con immagini che contengono dati discreti (come etichette in un'immagine di segmentazione).
#                         Svantaggi: Può portare a immagini ridimensionate di qualità inferiore, specialmente quando si ingrandisce, poiché i pixel vengono replicati e questo può rendere l'immagine "a blocchi" o pixelata.
# 2 ) ToLabel() --->     Converte l'immagine in un formato adatto per essere utilizzato come etichetta in un task di segmentazione. Questo potrebbe significare la conversione dell'immagine in un tensore e/o la modifica del tipo di dati in modo che sia adatto per rappresentare etichette categoriche
# 3 ) Relabel(255, 19) --> Cambia un particolare valore di etichetta nell'immagine. In questo caso, tutte le occorrenze del valore 255 vengono cambiate in 19. Il valore 255 è spesso usato nelle immagini di etichetta per rappresentare un 'etichetta di ignorare' o 'sfondo'. In questo contesto, il cambiamento in 19 potrebbe essere utilizzato per adattare le etichette al numero di classi previste dal modello o per una particolare convenzione di etichettatura.
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    print(f"Evaluation of mIoU using the metrics : {args.typeConfidence}")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    model2 = Net(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # La funzione load_my_state_dict nel codice che hai fornito è progettata per caricare i pesi (o parametri) di un modello di machine learning da un file salvato
    # La funzione accetta due argomenti: model, che è il modello di machine learning in cui si desidera caricare i pesi, e state_dict, che è un dizionario contenente i pesi da caricare.
    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        # Questa linea estrae lo stato attuale del modello, cioè i parametri attuali (pesi, bias, ecc.) del modello. state_dict() è una funzione PyTorch che restituisce un ordinato dizionario (OrderedDict) dei parametri.
        own_state = model.state_dict()
        # Il codice itera attraverso ogni coppia chiave-valore nel dizionario state_dict. name è il nome del parametro (per esempio, il nome di un particolare strato o peso nella rete), e param sono i valori effettivi dei pesi per quel nome.
        for name, param in state_dict.items():
            if "weight_orig" in name:
                # Il nome del parametro originale senza il suffisso '_orig'
                original_name = name.replace("_orig", "")
                # Recupera la maschera e il parametro originale
                mask = state_dict[name.replace("weight_orig", "weight_mask")]
                original_param = state_dict[name]
                # Applica la maschera al parametro
                pruned_param = original_param * mask
                name = original_name
                param = pruned_param
                # original_name = original_name.replace("module.","")
                # Aggiorna il parametro nel modello
                # getattr(model, original_name).data.copy_(pruned_param)
                # own_state[original_name].copy_(pruned_param)

            if "weight_mask" not in name:
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

        return model  # l'aggiornamento diretto dei pesi viene fatto copiando i nuovi valori dei pesi nei opportuni layer del own_state il quale tiene un puntatore dei vari layer (cosi la modifica si propaga subito anche al modello di partenza)

    # torch.load è una funzione in PyTorch che carica un oggetto salvato da un file. Questo oggetto può essere qualsiasi cosa che sia stata salvata precedentemente con torch.save, come un modello, un dizionario di stato del modello, un tensore, ecc.
    # map_location è un argomento di torch.load che specifica come e dove i tensori salvati devono essere mappati in memoria. Può essere utilizzato per forzare tutti i tensori ad essere caricati su CPU o su una specifica GPU, o per mapparli da una configurazione di hardware a un'altra.
    # Nel tuo esempio, map_location=lambda storage, loc: storage è una funzione lambda che ignora il loc (la localizzazione originale del tensore quando è stato salvato) e restituisce storage. Questo significa che i tensori saranno caricati sulla stessa tipologia di dispositivo da cui sono stati salvati (CPU o GPU).
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    model2 = load_my_state_dict(model2, torch.load( args.loadDir + "model_best.pth", map_location=lambda storage, loc: storage))
    argsPlus = {
        "listLayerPruning": ["non_bottleneck_1d"],
        "listNumLayerPruning": [],
        "listInnerLayerPruning": ["conv"],
        "pruning": 0.3,

    }
    argsPlus = SimpleNamespace(**argsPlus)
    model2 = myutils.remove_prunned_channels_from_model(model2, argsPlus)
    print ("Model and weights LOADED successfully")

    calibrationQuantization = DataLoader( cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # commando per forzare il modello ad essere in modelità valutazione
    model.eval()
    model2.eval()
    model2.quantize()

    for step, (images, labels, _, _) in enumerate(calibrationQuantization):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        model2.quantize_forward(images)

    model2.freeze()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)
    iouEvalVal2 = iouEvalVal(NUM_CHANNELS)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)
            outputs2 = model2.quantize_inference(images)

        finalOutput = outputs.max(1)[1].unsqueeze(1)

        if args.typeConfidence.casefold().replace(" ", "") == "msp":
            temperature = args.temperature
            scaledresult = outputs / temperature
            probs = torch.nn.functional.softmax(scaledresult, 1)  # result = model(images), F = torch.nn.functional
            _, predicted_classes = torch.max(probs, dim=1)
            finalOutput = predicted_classes.unsqueeze(1)
        if args.typeConfidence.casefold().replace(" ", "") == "maxentropy":
            eps = 1e-10
            probs = torch.nn.functional.softmax(outputs, dim=1)
            entropy = torch.div(torch.sum(-probs * torch.log(probs + eps), dim=1),torch.log(torch.tensor(probs.shape[1]) + eps))
            confidence = 1 - entropy
            weighted_output = probs * confidence.unsqueeze(1)
            _, predicted_classes = torch.max(weighted_output, dim=1)
            finalOutput = predicted_classes.unsqueeze(1)

        iouEvalVal.addBatch(finalOutput.data, labels)
        iouEvalVal2.addBatch(outputs2.max(1)[1].unsqueeze(1))

        filenameSave = filename[0].split("leftImg8bit/")[1]

        #print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()
    iouVal2, iou_classes2 = iouEvalVal2.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    iou_classes_str2 = []
    for i in range(iou_classes2.size(0)):
        iouStr = getColorEntry(iou_classes2[i]) + '{:0.2f}'.format(iou_classes2[i] * 100) + '\033[0m'
        iou_classes_str2.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

    print("---------------------------------------")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")
    # print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str2[0], "Road")
    print(iou_classes_str2[1], "sidewalk")
    print(iou_classes_str2[2], "building")
    print(iou_classes_str2[3], "wall")
    print(iou_classes_str2[4], "fence")
    print(iou_classes_str2[5], "pole")
    print(iou_classes_str2[6], "traffic light")
    print(iou_classes_str2[7], "traffic sign")
    print(iou_classes_str2[8], "vegetation")
    print(iou_classes_str2[9], "terrain")
    print(iou_classes_str2[10], "sky")
    print(iou_classes_str2[11], "person")
    print(iou_classes_str2[12], "rider")
    print(iou_classes_str2[13], "car")
    print(iou_classes_str2[14], "truck")
    print(iou_classes_str2[15], "bus")
    print(iou_classes_str2[16], "train")
    print(iou_classes_str2[17], "motorcycle")
    print(iou_classes_str2[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal2) + '{:0.2f}'.format(iouVal2 * 100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--typeConfidence', default='MaxLogit')
    parser.add_argument("--temperature", default = 1.0)

    if os.path.basename(os.getcwd()) != "eval":
        os.chdir("./eval")

    main(parser.parse_args())
