# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import shutil
import time
import numpy as np
import torch
import math
import gc

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler, optimizer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20  # pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


# Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512, backbone=None):
        self.enc = enc  # A flag (True/False) to enable additional processing on the target image.
        self.augment = augment  # A flag to enable or disable augmentation.
        self.height = height  # The desired height to resize images.
        self.backbone = backbone
        pass

    def __call__(self, input, target):  # method is executed when an instance of the class MyCoTransform is invoked

        # I due input contengono la stessa immagine di cui input è l'immagine di partenza mentre target è la stessa immagine dove ad ogni pixel è stato già stato
        # assegnato un label, in questo caso ogni pixel avrà un valore tra 0 ee 19 dove ognuno di essi rappresenta la classa assegnata al pixel (ovvero il pixel i-esimo con valore 1 indica che è stato predetto come veicolo, 2 come auto etc etc)

        # Resize strict the images to the target dimension self.height (define as parameter) and applies a transformation called (interpolazione)
        # L'interpolazione è una tecnica utilizzata per il ridimensionamento delle immagini e per altre trasformazioni geometriche
        # per calcolare i valori dei nuovi pixel basandosi sui pixel di partenza
        # if self.backbone.casefold().replace(" ", "") == "barlowtwins":
        # input = Resize((224, 224))(input)
        # else:
        input = Resize(self.height, Image.BILINEAR)(
            input)  # L'interpolazione bilineare (BILINEAR) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione bilineare considera i 4 pixel più vicini nella posizione corrispondente dell'immagine originale. Il valore del nuovo pixel è calcolato come una media ponderata dei valori di questi quattro pixel. Le ponderazioni sono basate sulla distanza relativa del punto calcolato rispetto a ciascuno di questi quattro pixel. In termini semplici, più un pixel è vicino al punto calcolato, maggiore sarà il suo contributo al valore finale.

        target = Resize(self.height, Image.NEAREST)(
            target)  # L'interpolazione nearest neighbor (Nearest) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione nearest neighbor semplicemente seleziona il valore del pixel più vicino nell'immagine originale, senza considerare altri pixel vicini. In altre parole, il valore del nuovo pixel è uguale a quello del pixel più vicino nella posizione corrispondente dell'immagine originale.

        if (self.augment):
            # Random hflip
            hflip = random.random()  # define randomly a value to chose if flip horizontal both images or not (specchiare l'immagine)
            if (hflip < 0.5):  # 50% di ruotare l'immagine e 50% no, per aumeentare randomicità nei dati. Per cui alcuni sono specchiati nella fase di augmentation altri no
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            # Both images are randomly shifted by 0-2 pixels in x and y directions
            # Ad esempio transX = 2, transY = 2 sposta l'immagine verso destra e in basso, mentre una traslazione negativa (ad esempio transX = -2, transY = -2) sposta l'immagine verso sinistra e in alto.
            transX = random.randint(-2,
                                    2)  # define randomly how much shift the images from 2 pixel to the left to 2 pixel to the right (could be also 0)
            transY = random.randint(-2,
                                    2)  # define randomly how much shift the images from 2 pixel to the bottom to 2 pixel to the up (could be also 0)

            # riga 66 e 67 servono per traslare l'immagine con i valori definiti da transX e transY e i pixel aggiunti vengono riempiti con logiche diverse a seconda se immagine input o immagine output
            input = ImageOps.expand(input, border=(transX, transY, 0, 0),
                                    fill=0)  # pad the input è stato riempito con 0 in quei pixel (questo significa che i pixel sono stati resi blu)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0),
                                     fill=255)  # pad label filling with 255 (questo significa che quei pixel sono stati resi bianchi)

            # serve proprio a ritagliare l'immagine traslata per riportarla alle sue dimensioni originali, ma con il contenuto dell'immagine spostato in base ai valori di transX e transY.
            # Questo ritaglio riduce le dimensioni dell'immagine traslata per farle corrispondere alle sue dimensioni originali. Tuttavia, a causa della traslazione precedente,
            # la parte visibile dell'immagine sarà ora differente rispetto all'originale. Ad esempio, se l'immagine era stata spostata verso destra e in basso, il ritaglio rimuoverà parti dell'immagine originale dal lato destro e inferiore.
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = normalize(input)

        if (self.enc):
            target = Resize(int(self.height / 8), Image.NEAREST)(
                target)  # avviene un resize probabilmente per portare l'immagine ad avere dimensioni che poi saranno usate per la fase di convoluzione
        target = ToLabel()(target)
        # l'operazione di Relabel consiste per l'output target ovvero quello che già ha una classificazione per ogni pixel, di cambiare tutti i pixel con valore 255 a valore 19
        # questo perchè magari pixel 255 non ha una label associata mentre 19 è la label per classificazione generica (es : background)
        target = Relabel(255, 19)(target)

        return input, target


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
            self.loss = torch.nn.NLLLoss(weight,ignore_index=ignores_index)
        else:
            self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):

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


def train(args, model, enc=False):
    best_acc = 0

    #torch.distributed.init_process_group(
    #    backend='nccl', init_method='tcp://localhost:58472',
    #    world_size=torch.cuda.device_count(), rank=0)

    # TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    # create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    # Ogni elemento nel vettore di pesi corrisponde a una delle categorie (o classi) nel tuo problema di classificazione o segmentazione. Per esempio, il primo peso potrebbe corrispondere alla classe "background", il secondo a "auto", il terzo a "pedone", e così via.
    # Se alcune di queste 20 classi sono meno frequenti o più critiche da identificare correttamente rispetto ad altre, puoi assegnare loro pesi maggiori nel vettore di pesi. Questo farà sì che la funzione di perdita "premi" il modello più fortemente per le previsioni corrette o "penalizzi"
    # più duramente per le previsioni errate in quelle classi, aiutando a bilanciare l'effetto delle classi sbilanciate durante l'addestramento.

    # Ogni peso nel vettore è staticamente associato a una specifica classe per tutta la durata dell'addestramento del modello.
    # Quando inizializzi il vettore di pesi, ciascun peso è assegnato a una specifica classe. Ad esempio, in un vettore di pesi di dimensione 20, il primo peso potrebbe essere associato alla prima classe, il secondo peso alla seconda classe, e così via. Questa associazione non cambia durante l'addestramento

    # In molti set di dati di classificazione, alcune classi possono essere molto più frequenti di altre. Ad esempio, in un set di dati di segmentazione stradale, la classe "strada" potrebbe essere molto più comune della classe "pedone".
    # Senza un adeguato bilanciamento, un modello di machine learning potrebbe diventare parziale verso le classi più frequenti, imparando principalmente a riconoscerle e ignorando o non performando bene sulle classi meno frequenti.
    # Assegnando pesi diversi alle diverse classi nella funzione di perdita, si può bilanciare l'importanza data ad ogni classe durante l'addestramento del modello. In generale, si assegna un peso maggiore alle classi meno frequenti e un peso minore alle classi più frequenti.
    # Questo aiuta a garantire che il modello non ignori le classi meno frequenti. Quando si calcola la perdita per una previsione, il valore della perdita viene moltiplicato per il peso associato alla classe vera di quel dato campione. Quindi, errori in classi con peso maggiore contribuiscono di più alla perdita totale,
    # il che spinge il modello a prestare maggiore attenzione a queste classi durante l'addestramento.
    # Se la classe "pedone" è rara nel set di dati ma è molto importante riconoscerla correttamente (ad esempio, per motivi di sicurezza nella guida autonoma), assegnandole un peso maggiore nella funzione di perdita, si può incentivare il modello a migliorare la sua capacità di rilevare pedoni, nonostante la loro relativa rarità nel set di dati.
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

    # classi per gestire augmentation per il dataset di training e quello di validation
    co_transform = MyCoTransform(enc, augment=False, height=args.height,backbone=args.backbone)  # 1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height,backbone=args.backbone)  # 1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
    # print(type(criterion))

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

    # Adam aggiorna i pesi della rete neurale in direzione opposta rispetto al gradiente della funzione di perdita. Questo aiuta a minimizzare la funzione di perdita. Ora ci concentriamo sui vari parametri passati :
    # 1.  model.parameters() --> Questo argomento passa tutti i parametri addestrabili del tuo modello all'ottimizzatore. In pratica, questi sono i pesi e i bias della rete neurale che Adam aggiornerà durante il processo di addestramento.
    # 2.  learning rate (lr) --> Il learning rate controlla quanto velocemente il modello apprende; un valore troppo alto può far sì che l'apprendimento sia instabile, mentre un valore troppo basso può portare a un apprendimento molto lento.
    # 3.  betas --> Questi sono i valori per i parametri beta1 e beta2 di Adam. Gestiscono rispettivamente il decadimento esponenziale dei tassi medi del gradiente passato e del quadrato del gradiente passato. beta1 e beta2 in Adam sono fondamentali per bilanciare la quantità di informazioni del passato (gradienti passati e la variazione di questi gradienti) che vengono incluse negli aggiornamenti attuali dei parametri della rete.
    # 3a. beta1 --> controlla la media mobile esponenziale del primo momento, cioè la media dei gradienti. In termini semplici, tiene traccia della direzione e della velocità con cui i parametri della rete neurale stanno cambiando. Questa media aiuta a smorzare le oscillazioni dei gradienti e a indirizzare l'ottimizzazione in modo più stabile e consistente verso il minimo della funzione di perdita. Il valore 0.9 per beta1 significa che il momento attuale tiene conto principalmente del gradiente attuale, ma include anche una frazione significativa della storia dei gradienti precedenti.
    # 3b. beta2 --> controlla la media mobile esponenziale del secondo momento, cioè la media dei quadrati dei gradienti. Questa media aiuta a regolare la grandezza degli aggiornamenti dei pesi in base alla varianza dei gradienti. In pratica, permette all'ottimizzatore di adattarsi alla scala di ciascun parametro, rendendo l'addestramento più efficiente e stabile, specialmente in presenza di gradienti rumorosi o di diverse scale tra i parametri. Il valore 0.999 per beta2 implica che l'aggiornamento dei pesi tiene in considerazione una finestra più lunga di gradienti passati, fornendo una stima più stabile e meno rumorosa della varianza dei gradienti.
    # 4.  Epsilon (eps) -->  è un piccolo valore aggiunto per migliorare la stabilità numerica dell'algoritmo. Aiuta a prevenire la divisione per zero durante l'aggiornamento dei parametri.
    # 5.  weight_decay -->  Il weight decay è un metodo di regolarizzazione che aiuta a prevenire l'overfitting riducendo leggermente i valori dei pesi ad ogni iterazione.

    optimizer = Adam(model.parameters(), 5e-8, (0.9, 0.999), eps=1e-08, weight_decay=5e-5)  ## scheduler 1
    #optimizer = torch.optim.AdamW(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08,weight_decay=1e-4)  ## scheduler 2
    # optimizer = torch.optim.SGD(model.parameters(),  5e-4, momentum=0.9, weight_decay=1e-4)

    # Uno scheduler del learning rate è utilizzato per modificare il learning rate durante il processo di addestramento, secondo una certa politica.
    # Ad ogni epoca durante l'addestramento, lo scheduler aggiusterà il learning rate moltiplicandolo per il valore restituito dalla funzione lambda1.
    # Ciò significa che man mano che l'addestramento procede e si avvicina al numero totale di epoche, il learning rate diminuirà seguendo la legge definita nella funzione lambda.

    # lambda1 --> Questa è una funzione lambda in Python che prende come input l'epoca corrente (epoch) e calcola un fattore di scala per il learning rate
    # La formula pow(...) riduce gradualmente il learning rate durante il processo di addestramento. All'inizio dell'addestramento (epoch vicino a 0), questo valore è vicino a 1, quindi il learning rate rimane quasi invariato.
    # Man mano che l'addestramento procede e epoch aumenta, il valore ritorna da questa funzione diminuisce, riducendo così il learning rate.

    # scheduler --> Questa istruzione crea un oggetto scheduler di tipo LambdaLR (un tipo di scheduler del learning rate che permette di regolare il learning rate in base a una funzione definita dall'utente)
    # riceve come parametri :
    # optimizer --> è l'ottimizzatore per il quale stai regolando il learning rate (Adam).
    # lr_lambda -->  è un parametro che accetta una funzione o una lista di funzioni. Queste funzioni sono usate per regolare il learning rate

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)  ## scheduler 2
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=20, mode='triangular', cycle_momentum=False)  # scheduler 3

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
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))
        del checkpoint
        gc.collect()

    if not args.resume and args.backbone:
        model.loadInitialWeigth("../save/checkpoint_barlowTwins.pth")

    # se sono stati impostati visualize a True ed è stato settato una cardinalità per mostrare la visualizzazione ogni tot step ( step rappresenta essenzialmente il numero del batch corrente durante l'iterazione del DataLoader)
    # In caso positivo viene creata un istanza di Dashboard che al suo interno ha metodi per visualizzare perdite e immagini.
    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    if args.freezingBackbone:
      print("Freezing the backbone ... ")
      # Congela i pesi dell'encoder
      if isinstance(model, torch.nn.DataParallel):
        for param in model.module.encoder.parameters():
          param.requires_grad = False
      else:
        for param in model.encoder.parameters():
          param.requires_grad = False
    else:
      print("Not freezing the backbone ... ")

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

        # serve a impostare il modello in modalità "training", ovvero viene comunicato a tutti i layer del modello che ora si trovano nella fase di addestramento. Questo è importante perché alcuni layer possono comportarsi diversamente a seconda che il modello sia in fase di addestramento o di valutazione.
        # Ecco cosa viene fatto nel dettaglio (model.eval al contrario setta il modello in modalità "valutazione") :
        # 1. Dropout: Durante l'addestramento, il dropout "disattiva" casualmente alcuni neuroni (o connessioni tra neuroni) per prevenire l'overfitting e promuovere un apprendimento più robusto. In fase di valutazione, il dropout viene disabilitato per utilizzare l'intera rete per le previsioni.
        # 2. Batch Normalization: Durante l'addestramento, la batch normalization normalizza l'output di un layer utilizzando la media e la varianza del batch corrente. Durante la valutazione, invece, utilizza la media e la varianza calcolate dall'intero set di addestramento.
        model.train()
        # model.to(torch.float64)

        # for name, param in model.named_parameters():
        # if param.requires_grad:
        # print(f'Weight stats before optimization - {name}: mean={param.data.mean()}, std={param.data.std()}')

        # La funzione enumerate in Python è un modo conveniente per ottenere sia l'indice sia i valori da un iteratore.
        # In questo caso, enumerate(loader) restituisce due valori ad ogni iterazione: un indice (che viene assegnato a step) e i valori (in questo caso, tuple (images, labels)). Dove :
        # step --> In ogni iterazione del ciclo for, step assume il valore dell'indice corrente fornito da enumerate. Inizia da 0 e si incrementa di 1 ad ogni iterazione. Quindi, step rappresenta essenzialmente il numero del batch corrente durante l'iterazione del DataLoader.
        # (images, labels) --> tupla di dim batch_size. Ovvero al suo interno ha batch_size coppie di immagine e relativo ground truth
        for step, (images,images2, labels) in enumerate(loader):

            start_time = time.time()
            # print (labels.size())
            # print (np.unique(labels.numpy()))
            # print("labels: ", np.unique(labels[0].numpy()))
            # labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                images = images2.cuda()

            # Variable era una classe fondamentale utilizzata per incapsulare i tensori e fornire la capacità di calcolo automatico del gradiente (autograd).
            # Quando si avvolgeva un tensore in un oggetto Variable, si permetteva a PyTorch di tracciare automaticamente tutte le operazioni eseguite su di esso e
            # calcolare i gradienti durante la backpropagation.Da PyTorch 0.4 in poi, la funzionalità di Variable è stata integrata direttamente nei tensori, ora, ogni tensore ha un attributo requires_grad che, se impostato su True, abilita il calcolo del gradiente per quel tensore in modo simile a come funzionavano le Variable.
            images.requires_grad_(True)
            inputs = images

            images2.requires_grad_(True)
            inputs2 = images2

            # labels.requires_grad_(True)
            targets = labels

            if "BiSeNet" in args.model:
                outputs = model(inputs)
                outputs = outputs[0]
                outputs = outputs.float()
            if "erfnet" in args.model:
                outputs = model(inputs,inputs2, only_encode=enc)

            # print("targets", np.unique(targets[:, 0].cpu().data.numpy()))
            # Prima di calcolare i gradienti per l'epoca corrente, è necessario azzerare i gradienti accumulati dalla bacth precedente.
            # Questo è essenziale perché, per impostazione predefinita, i gradienti si sommano in PyTorch per consentire l'accumulo di gradienti in più passaggi.
            optimizer.zero_grad()

            # Viene calcolata la perdita (o errore) utilizzando la funzione di perdita definita da CrossEntropyLoss2d per misurare la differenza tra le previsioni del modello (outputs) e le etichette vere (targets).
            # targets[:, 0] suggerisce che stai selezionando una specifica colonna o una parte specifica delle etichette target (?).
            loss = criterion(outputs, targets[:, 0])

            # Questo calcola i gradienti della perdita rispetto ai parametri del modello. È il passo in cui il modello "impara", aggiornando i gradienti in modo da minimizzare la perdita.
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Questo passaggio aggiorna i pesi del modello utilizzando i gradienti calcolati nel passaggio backward. L'ottimizzatore Adam (definito sopra) modifica i pesi per minimizzare la perdita.
            optimizer.step()

            # stai essenzialmente dicendo allo scheduler di calcolare e impostare il nuovo learning rate basandosi sull'epoca corrente.
            # La funzione lambda o la logica definita nello scheduler determina come il learning rate dovrebbe cambiare a quella specifica epoca.
            scheduler.step(loss.item())  ## scheduler 2

            # epoch_loss è un vettore in cui sono aggiunti ad ogni batch il valore ritornato dalla loss function
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                # start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                # print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            # print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                # image[0] = image[0] * .229 + .485
                # image[1] = image[1] * .224 + .456
                # image[2] = image[2] * .225 + .406
                # print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
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
            # Con l'integrazione delle funzionalità di Variable direttamente nei tensori in PyTorch 0.4 e versioni successive, l'uso di volatile è stato deprecato.
            # Al suo posto, PyTorch ha introdotto un modo più intuitivo e meno soggetto a errori per gestire il calcolo dei gradienti: il contesto with torch.no_grad():
            # Quando si eseguono operazioni all'interno di un blocco with torch.no_grad():, PyTorch non traccia, calcola o memorizza gradienti. Questo riduce l'uso della memoria e accelera i calcoli quando i gradienti non sono necessari, come appunto durante l'inferenza o il test.
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

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)

                # Add batch to calculate TP, FP and FN for iou estimation
                if (doIouVal):
                    # start_time_iou = time.time()
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                    # print ("Time to add confusion matrix: ", time.time() - start_time_iou)

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
        # scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
            print("EPOCH IoU on VAL set: ", iouStr, "%")

            # remember best valIoU and save checkpoint
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
            filenamebest = f'{savedir}/model_best.pth'
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
            modelFilenameDrive = args.model + ("FreezingBackbone" if args.freezingBackbone else "")
            saveOnDrive(epoch = epoch , model = modelFilenameDrive, pathOriginal = f"/content/AMLProjectBase/save/{args.savedir}/")



    return (model)  # return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

def saveOnDrive(epoch , model , pathOriginal):
    if not os.path.isdir(pathOriginal):
        print(f"Path Original is wrong : {pathOriginal}")
    drive = "/content/drive/MyDrive/"
    if os.path.isdir(drive):
        if not os.path.isdir(drive+f"AML/"):
            os.mkdir(drive+f"AML/")
        if not os.path.exists(drive + f"AML/{model}/"):
            os.mkdir(drive + f"AML/{model}/")
        shutil.copy2(pathOriginal+"/checkpoint.pth.tar", drive + f"AML/{model}/checkpoint.pth.tar")
        shutil.copy2(pathOriginal + "/automated_log.txt", drive + f"AML/{model}/automated_log.txt")
        shutil.copy2(pathOriginal + "/opts.txt", drive + f"AML/{model}/opts.txt")
        shutil.copy2(pathOriginal + "/model.txt", drive + f"AML/{model}/model.txt")
        if os.path.isfile(pathOriginal + "/model_best.pth"):
            shutil.copy2(pathOriginal + "/model_best.pth", drive + f"AML/{model}/model_best.pth")
        print(f"Checkpoint of epoch {epoch} saved on Drive path : {drive}AML/{model}/")
    else:
        print("Drive is not linked ...")

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
    if "erfnet" in args.model and args.backbone:
        model = model_file.Net(NUM_CLASSES,encoder=None, batch_size = args.batch_size,backbone = args.backbone)
    if "erfnet" in args.model and not args.backbone:
        model = model_file.Net(NUM_CLASSES,encoder=None)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
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

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder

    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    # train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder
    # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0.
    # We must reinit decoder weights or reload network passing only encoder in order to train decoder
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
            pretrainedEnc = next(model.children()).encoder
        if args.model.casefold().replace(" ", "") == "erfnetbarlowtwins":
            model = model_file.Net(NUM_CLASSES, encoder=None, batch_size=args.batch_size,backbone=args.backbone)
        if args.model.casefold().replace(" ", "") == "erfnetbarlowtwinsloss":
            model = model_file.Net(NUM_CLASSES, encoder=None, batch_size=args.batch_size,backbone=args.backbone)
        if args.model.casefold().replace(" ", "") == "BiSeNet":
            model = model_file.BiSeNetV1(NUM_CLASSES, 'train')
        if args.model.casefold().replace(" ", "") == "erfnet" :
            model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  # Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)  # Train decoder
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=False)  # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
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
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder')  # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize',
                        action='store_true')  # variabile per determinare se la visualizzazione è attivata o meno.

    parser.add_argument('--iouTrain', action='store_true',  # boolean to compute IoU evaluation  in the training phase
                        default=False)  # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true',
                        default=True)  # boolean to compute IoU evaluation also in the validation phase
    parser.add_argument('--resume', action='store_true')  # Use this flag to load last checkpoint for training
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument("--freezingBackbone",action='store_true')
    parser.add_argument("--saveCheckpointDriveAfterNumEpoch",type=int, default=1)

    if os.path.basename(os.getcwd()) != "train":
        os.chdir("./train")

    main(parser.parse_args())
