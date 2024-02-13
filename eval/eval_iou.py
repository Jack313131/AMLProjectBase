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
from torch.optim import SGD, Adam, lr_scheduler, optimizer
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

import utils as myutils
from dataset import cityscapes
# from erfnet import ERFNet
from erfnet import Net
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
    Relabel(255, 19),  # ignore label to 19
])


class CrossEntropyLoss2d(torch.nn.Module):
    # è progettata per lavorare con input bidimensionali (bidimensionali si intende con più di una dimensione). In contesti come la segmentazione delle immagini, ogni pixel dell'immagine viene classificato in una delle categorie.
    # Pertanto, l'output del modello e il target (ground truth) sono immagini bidimensionali, dove ogni pixel ha una classe associata.
    # Pertanto, l'output di un modello di segmentazione delle immagini non è semplicemente un'immagine 2D, ma un tensore con dimensioni [batch_size, numero_di_classi, altezza, larghezza].
    # In questo modo, il modello può prevedere la classe di ciascun pixel per ogni immagine nel batch.
    # CrossEntropyLoss2d è ideale per questo scopo perché calcola una perdita per ogni pixel dell'immagine, basandosi su quanto bene il modello predice la classe di quel pixel rispetto alla classe effettiva.

    def __init__(self, weight=None, ignores_index=None):
        super().__init__()

        # Questo inizializza la Negative Log Likelihood Loss in 2D (NLLLoss2d), questa è una funzione di perdita utilizzata in problemi di classificazione e segmentazione delle immagini, dove l'obiettivo è classificare ciascun pixel dell'immagine
        # NLLLoss lavora con log-probabilità, non con probabilità dirette. Si presume che l'output del modello (le previsioni) siano log-probabilità di ciascuna classe. Queste log-probabilità sono tipicamente ottenute applicando la funzione log_softmax ai logits (l'output grezzo del modello prima dell'applicazione di una funzione di attivazione come softmax).
        # Nella segmentazione delle immagini, hai una mappa di etichettatura (o immagine target) dove ogni pixel ha una etichetta di classe assegnata. La NLLLoss in 2D calcola la perdita per ogni pixel individualmente. Per un dato pixel, la perdita è il negativo del logaritmo della probabilità predetta per la classe vera di quel pixel. Matematicamente, per un pixel con classe vera c,
        # se pc è la probabilità logaritmica predetta per quella classe, la perdita per quel pixel è -pc
        if ignores_index is not None:
            self.loss = torch.nn.NLLLoss(weight, ignore_index=ignores_index)
        else:
            self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):

        # targets = targets.long()

        # Converti i outputs in float32 per la computazione della loss
        # outputs = outputs.float()

        # if targets.dtype != torch.long:
        # targets = targets.long()

        # outputs = torch.clamp(outputs, min=-3, max=3)
        # reg_term = 0.1 * torch.norm(outputs, p=2, dim=1).mean()

        # outputs = torch.clamp(outputs, min=-5, max=5)
        # outputs = outputs.to(torch.float64)
        # targets = targets.double()
        # outputs sono le previsioni date dal modello (output sono in forma di logits, ossia valori grezzi non normalizzati, per ciascuna classe e per ogni pixel.), mentre target il ground truth (Queste sono le etichette vere che si desidera che il modello impari a predire.)
        # prima di passare al confronto tra previsioni del modello e ground truth, la predizione del modello viene passata ad una softmax per ottenere un vettore di probabilità, dove ogni valore rappresenta la probabilità che un dato pixel appartenga a una particolare classe.
        # Poi, self.loss, che è la funzione NLLLoss2d, viene applicata per calcolare la perdita effettiva. Questa funzione calcola la log likelihood negativa tra le previsioni (dopo aver applicato log_softmax) e le etichette vere.
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)  # + reg_term


def main(args):
    weight = torch.ones(NUM_CLASSES)

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

    print(f"Evaluation of mIoU using the metrics : {args.typeConfidence}")

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    if args.loadWeightsPruned:
        path_model_mod = args.loadDir + args.loadWeightsPruned
    if args.loadModelPruned:
        path_model_mod = args.loadDir + args.loadModelPruned

    print("Loading model Original: " + modelpath)
    print("Loading weights Original: " + weightspath)


    modelOriginal = Net(NUM_CLASSES)
    modelMod = Net(NUM_CLASSES)

    # modelOriginal = torch.nn.DataParallel(modelOriginal)
    if (not args.cpu):
        modelOriginal = torch.nn.DataParallel(modelOriginal).cuda()

    # La funzione load_my_state_dict nel codice che hai fornito è progettata per caricare i pesi (o parametri) di un modello di machine learning da un file salvato
    # La funzione accetta due argomenti: modelOriginal, che è il modello di machine learning in cui si desidera caricare i pesi, e state_dict, che è un dizionario contenente i pesi da caricare.
    def load_my_state_dict(model, state_dict):  # custom function to load modelOriginal when not all dict elements
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
    # torch.load è una funzione in PyTorch che carica un oggetto salvato da un file. Questo oggetto può essere qualsiasi cosa che sia stata salvata precedentemente con torch.save, come un modello, un dizionario di stato del modello, un tensore, ecc.
    # map_location è un argomento di torch.load che specifica come e dove i tensori salvati devono essere mappati in memoria. Può essere utilizzato per forzare tutti i tensori ad essere caricati su CPU o su una specifica GPU, o per mapparli da una configurazione di hardware a un'altra.
    # Nel tuo esempio, map_location=lambda storage, loc: storage è una funzione lambda che ignora il loc (la localizzazione originale del tensore quando è stato salvato) e restituisce storage. Questo significa che i tensori saranno caricati sulla stessa tipologia di dispositivo da cui sono stati salvati (CPU o GPU).
    modelOriginal = load_my_state_dict(modelOriginal, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights Original LOADED successfully")

    print("Loading model and weights Mod: " + path_model_mod)

    modelMod = myutils.remove_mask_from_model_with_pruning(modelMod, torch.load(path_model_mod,
                                                   map_location=lambda storage, loc: storage),path_model_mod)
    myutils.save_model_mod_on_drive(modelMod)
    print("Model and weights Mod LOADED successfully")

    calibrationQuantization = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # commando per forzare il modello ad essere in modelità valutazione
    modelMod.train()

    if torch.cuda.is_available():
        modelMod = modelMod.to("cuda")
        weight = weight.cuda()

    criterion = CrossEntropyLoss2d(weight)
    optimizer = Adam(modelOriginal.parameters(), 5e-8, (0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)

    print("Start fine tuning pruning for adding layers ... ")
    print("Freezing layer not named : adaptingInput")
    for name, param in modelMod.named_parameters():
        # Congela i parametri se 'adaptingInput' non è nel nome del layer
        if 'adaptingInput' not in name:
            param.requires_grad = False
        else:
            # Assicurati che i parametri che non devono essere congelati siano settati per il gradiente
            param.requires_grad = True

    for epoch in range(1, 15):
        print("----- TRAINING - EPOCH", epoch, "-----")
        epoch_loss = []
        time_train = []
        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        modelOriginal.train()

        for step, (images, labels, _, _) in enumerate(calibrationQuantization):

            start_time = time.time()

            # Prima di calcolare i gradienti per l'epoca corrente, è necessario azzerare i gradienti accumulati dalla bacth precedente.
            # Questo è essenziale perché, per impostazione predefinita, i gradienti si sommano in PyTorch per consentire l'accumulo di gradienti in più passaggi.
            optimizer.zero_grad()

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            images.requires_grad_(True)
            inputs = images

            # labels.requires_grad_(True)
            targets = labels

            outputsOriginal = modelMod(inputs)
            # print("Outputs: ", outputsOriginal.shape)

            loss = criterion(outputsOriginal, targets[:, 0])
            optimizer.zero_grad()
            # Questo calcola i gradienti della perdita rispetto ai parametri del modello. È il passo in cui il modello "impara", aggiornando i gradienti in modo da minimizzare la perdita.
            loss.backward()

            optimizer.step()

            scheduler.step()  ## scheduler 2
            # epoch_loss è un vettore in cui sono aggiunti ad ogni batch il valore ritornato dalla loss function
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
            if step % 50 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))


    print("Model Pruned and Quantized ... ")

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalValOriginal = iouEval(NUM_CLASSES)
    iouEvalValMod = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputsOriginal = modelOriginal(inputs)
            outputsMod = modelMod(inputs)
            # outputsMod = modelMod.quantize_inference(inputs)

        finalOutput = outputsOriginal.max(1)[1].unsqueeze(1)

        if args.typeConfidence.casefold().replace(" ", "") == "msp":
            temperature = args.temperature
            scaledresult = outputsOriginal / temperature
            probs = torch.nn.functional.softmax(scaledresult, 1)  # result = modelOriginal(images), F = torch.nn.functional
            _, predicted_classes = torch.max(probs, dim=1)
            finalOutput = predicted_classes.unsqueeze(1)
        if args.typeConfidence.casefold().replace(" ", "") == "maxentropy":
            eps = 1e-10
            probs = torch.nn.functional.softmax(outputsOriginal, dim=1)
            entropy = torch.div(torch.sum(-probs * torch.log(probs + eps), dim=1),
                                torch.log(torch.tensor(probs.shape[1]) + eps))
            confidence = 1 - entropy
            weighted_output = probs * confidence.unsqueeze(1)
            _, predicted_classes = torch.max(weighted_output, dim=1)
            finalOutput = predicted_classes.unsqueeze(1)

        iouEvalValOriginal.addBatch(finalOutput.data, labels)
        iouEvalValMod.addBatch(outputsMod.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1]

        # print (step, filenameSave)

    iouValOriginal, iou_classes_original = iouEvalValOriginal.getIoU()
    iouValMod, iou_classes_mod = iouEvalValMod.getIoU()

    iou_classes_str_original = []
    for i in range(iou_classes_original.size(0)):
        iouStr = getColorEntry(iou_classes_original[i]) + '{:0.2f}'.format(iou_classes_original[i] * 100) + '\033[0m'
        iou_classes_str_original.append(iouStr)

    iou_classes_str_mod = []
    for i in range(iou_classes_mod.size(0)):
        iouStr = getColorEntry(iou_classes_mod[i]) + '{:0.2f}'.format(iou_classes_mod[i] * 100) + '\033[0m'
        iou_classes_str_mod.append(iouStr)

    # Assicurati che le variabili iou_classes_str_original e iou_classes_str_mod siano definite
    features_model_changed = path_model_mod.split("-")[2:]
    num_layers = str(features_model_changed[features_model_changed.index('Layer'):])
    text_model = (f"The model is with pruning {features_model_changed[0]} (amount : {features_model_changed[1]} & norm = {features_model_changed[4]}) "
                  f"for the modules {features_model_changed[5]} applied on layers {num_layers}")

    dir_model = path_model_mod.replace(".pth","")
    dir_save_result = f"{myutils.args.path_drive}Models/{dir_model}/results.txt"
    print(f"Saving result on path : {dir_save_result}")
    # Apertura (o creazione se non esiste) del file in modalità di scrittura
    with open(dir_save_result, 'w') as file:
        myutils.myutils.print_and_save("---------------------------------------", file)
        myutils.print_and_save(f"Took {time.time() - start} seconds", file)
        myutils.print_and_save("=======================================", file)

        # Il tuo codice commentato
        # myutils.print_and_save(f"TOTAL IOU: {iou * 100}%", file)

        myutils.print_and_save("Per-Class IoU:", file)
        for i in range(len(iou_classes_str_original)):
            myutils.print_and_save(
                f"{iou_classes_str_original[i]} (ModelOriginal) - {iou_classes_str_mod[i]} (Model Pruned) -- [Category Name]",
                file)

        myutils.print_and_save("=======================================", file)
        iouStr = getColorEntry(iouValOriginal) + '{:0.2f}'.format(iouValOriginal * 100) + '\033[0m'
        iouModStr = getColorEntry(iouValMod) + '{:0.2f}'.format(iouValMod * 100) + '\033[0m'
        myutils.print_and_save(f"MEAN IoU: {iouStr}% (Model Original) --- MEAN IoU: {iouModStr}%", file)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadWeightsPruned', default="")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=not torch.cuda.is_available())
    parser.add_argument('--typeConfidence', default='MaxLogit')
    parser.add_argument("--temperature", default=1.0)

    myutils.set_args(parser.parse_args())

    main(parser.parse_args())
