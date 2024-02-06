import torch
import torch.nn as nn
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
import torch.nn.init as init
import torch.nn.functional as F
import random
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
import os


class MyCoTransform(object):
    def __init__(self, augment=True, height=224):
        self.augment = augment  # A flag to enable or disable augmentation.
        self.height = height  # The desired height to resize images.
        pass

    def __call__(self, image1, image2):  # method is executed when an instance of the class MyCoTransform is invoked

        tipo_immagine = type(image1)
        print(image1.size())
        if tipo_immagine == Image.Image:
            print("L'immagine è un oggetto PIL")
        else:
            print("L'immagine non è un oggetto PIL")

        # I due input contengono la stessa immagine di cui input è l'immagine di partenza mentre target è la stessa immagine dove ad ogni pixel è stato già stato
        # assegnato un label, in questo caso ogni pixel avrà un valore tra 0 ee 19 dove ognuno di essi rappresenta la classa assegnata al pixel (ovvero il pixel i-esimo con valore 1 indica che è stato predetto come veicolo, 2 come auto etc etc)

        # Resize strict the images to the target dimension self.height (define as parameter) and applies a transformation called (interpolazione)
        # L'interpolazione è una tecnica utilizzata per il ridimensionamento delle immagini e per altre trasformazioni geometriche
        # per calcolare i valori dei nuovi pixel basandosi sui pixel di partenza
        image1 = Resize(self.height, Image.BILINEAR)(
            image1)  # L'interpolazione bilineare (BILINEAR) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione bilineare considera i 4 pixel più vicini nella posizione corrispondente dell'immagine originale. Il valore del nuovo pixel è calcolato come una media ponderata dei valori di questi quattro pixel. Le ponderazioni sono basate sulla distanza relativa del punto calcolato rispetto a ciascuno di questi quattro pixel. In termini semplici, più un pixel è vicino al punto calcolato, maggiore sarà il suo contributo al valore finale.
        image2 = Resize(self.height, Image.NEAREST)(
            image2)  # L'interpolazione nearest neighbor (Nearest) Per ogni nuovo pixel nell'immagine ridimensionata, l'interpolazione nearest neighbor semplicemente seleziona il valore del pixel più vicino nell'immagine originale, senza considerare altri pixel vicini. In altre parole, il valore del nuovo pixel è uguale a quello del pixel più vicino nella posizione corrispondente dell'immagine originale.

        if (self.augment):
            # Random hflip
            hflip = random.random()  # define randomly a value to chose if flip horizontal both images or not (specchiare l'immagine)
            if (
                    hflip < 0.5):  # 50% di ruotare l'immagine e 50% no, per aumeentare randomicità nei dati. Per cui alcuni sono specchiati nella fase di augmentation altri no
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            # Both images are randomly shifted by 0-2 pixels in x and y directions
            # Ad esempio transX = 2, transY = 2 sposta l'immagine verso destra e in basso, mentre una traslazione negativa (ad esempio transX = -2, transY = -2) sposta l'immagine verso sinistra e in alto.
            transX = random.randint(-2,
                                    2)  # define randomly how much shift the images from 2 pixel to the left to 2 pixel to the right (could be also 0)
            transY = random.randint(-2,
                                    2)  # define randomly how much shift the images from 2 pixel to the bottom to 2 pixel to the up (could be also 0)

            # riga 66 e 67 servono per traslare l'immagine con i valori definiti da transX e transY e i pixel aggiunti vengono riempiti con logiche diverse a seconda se immagine input o immagine output
            image1 = ImageOps.expand(image1, border=(transX, transY, 0, 0),
                                     fill=0)  # pad the input è stato riempito con 0 in quei pixel (questo significa che i pixel sono stati resi blu)
            image2 = ImageOps.expand(image2, border=(transX, transY, 0, 0),
                                     fill=255)  # pad label filling with 255 (questo significa che quei pixel sono stati resi bianchi)

            # serve proprio a ritagliare l'immagine traslata per riportarla alle sue dimensioni originali, ma con il contenuto dell'immagine spostato in base ai valori di transX e transY.
            # Questo ritaglio riduce le dimensioni dell'immagine traslata per farle corrispondere alle sue dimensioni originali. Tuttavia, a causa della traslazione precedente,
            # la parte visibile dell'immagine sarà ora differente rispetto all'originale. Ad esempio, se l'immagine era stata spostata verso destra e in basso, il ritaglio rimuoverà parti dell'immagine originale dal lato destro e inferiore.
            image1 = image1.crop((0, 0, image1.size[0] - transX, image1.size[1] - transY))
            image2 = image2.crop((0, 0, image2.size[0] - transX, image2.size[1] - transY))

        image1 = ToTensor()(image1)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image1 = normalize(image1)
        image2 = ToTensor()(image2)
        image2 = normalize(image2)

        return image1, image2


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):

    def __init__(self, batch_size, finalSize="8192-8192-8192", lambd=0.0051):
        super().__init__()
        self.finalSize = finalSize
        self.batch_size = batch_size
        self.lambd = lambd
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        sizes = [2048] + list(map(int, finalSize.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    # self.projector = nn.Sequential(*layers)

    def forward(self, x1,x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return z1*loss

        return x
