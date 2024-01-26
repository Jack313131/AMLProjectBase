# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

#La classe iouEval è progettata per calcolare l'Intersection over Union (IoU), una metrica comune per valutare la performance in compiti di segmentazione semantica.
class iouEval:

    # Inizializza la classe con un numero specificato di classi (nClasses) e un indice da ignorare (ignoreIndex).
    # Se ignoreIndex è maggiore del numero di classi, viene impostato su -1, il che significa che non ci sono indici da ignorare.
    # Chiama il metodo reset per inizializzare le variabili di conteggio.
    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    # Inizializza tre tensori:
    # self.tp (True Positives),
    # self.fp (False Positives) e
    # self.fn (False Negatives),
    # ognuno delle dimensioni pari al numero di classi (meno uno, se si considera un indice da ignorare).
    # Questi tensori sono usati per accumulare conteggi globali su più batch di dati.
    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    # Il metodo addBatch è un componente critico per calcolare le metriche di Intersection over Union (IoU) in compiti di segmentazione semantica.
    # Parametri di Input :
    # x: Predizioni del modello per un batch di immagini. Ha dimensioni [batch_size, nClasses, H, W], dove nClasses è il numero di classi di segmentazione, H e W sono rispettivamente altezza e larghezza dell'immagine.
    # y: Etichette di ground truth corrispondenti a x. Ha le stesse dimensioni di x.
    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)

        # Controllo della Disponibilità sulla GPU: Se x o y sono su una GPU, assicura che entrambi siano trasferiti sulla GPU per eseguire le operazioni.
        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # L'istruzione if (x.size(1) == 1) controlla effettivamente se il tensore x ha solo un canale lungo la sua seconda dimensione (C). Se questo è vero, significa che x è nel formato di classi indicizzate e ogni pixel ha un singolo valore di classe.
        # Se x ha più di un canale, si presume che sia già in un formato adatto per l'analisi successiva, come una rappresentazione one-hot o una distribuzione di probabilità.
        # Questo perchè l output del modello potrebbe essere di due tipologie :
        # 1. Formato di Classi Indicizzate: Ogni pixel ha un singolo valore intero che rappresenta la classe predetta. In questo caso, il tensore delle previsioni avrà una forma [batch_size, 1, H, W], dove 1 indica che c'è un solo canale, e ogni elemento di questo canale è l'indice della classe.
        # 2. Formato One-Hot o Distribuzione di Probabilità: Ogni pixel ha un vettore di valori, ciascuno dei quali rappresenta la probabilità (o il punteggio) che quel pixel appartenga a una determinata classe. In questo caso, il tensore avrà una forma [batch_size, nClasses, H, W], dove nClasses è il numero di classi possibili.
        # Vogliamo che il risultato del modello segui la seconda tipologia, ovvero una distribuzione di probabilità per ogni pixel. Se così non fosse vanno applicatee le opportune trasformazioni
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        # Filtra l'output del modello e il ground truth con le classi che possono essere intenzionalmente ignorate durante il calcolo.
        # if (self.ignoreIndex != -1): Questa istruzione verifica se esiste un indice specifico delle classi da ignorare (indicato da self.ignoreIndex).
        # Un valore di -1 per self.ignoreIndex indica che non ci sono classi da ignorare.
        if (self.ignoreIndex != -1):
            # Il tensore ignores indica le posizioni di ogni immagine nel batch dove la classe da ignorare è presente.
            # Questa informazione è utilizzata successivamente per escludere tali posizioni dai calcoli nell'analisi di segmentazione, in modo che la presenza della classe ignorata non influisca sulle metriche di valutazione come l'Intersection over Union (IoU).
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze() 

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou         #returns "iou mean", "iou per class"

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

