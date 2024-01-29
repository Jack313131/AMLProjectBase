# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch


# La classe iouEval è progettata per calcolare l'Intersection over Union (IoU), una metrica comune per valutare la performance in compiti di segmentazione semantica.
class iouEval:

    # Inizializza la classe con un numero specificato di classi (nClasses) e un indice da ignorare (ignoreIndex).
    # Se ignoreIndex è maggiore del numero di classi, viene impostato su -1, il che significa che non ci sono indici da ignorare.
    # Chiama il metodo reset per inizializzare le variabili di conteggio.
    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    # Inizializza tre tensori:
    # self.tp (True Positives),
    # self.fp (False Positives) e
    # self.fn (False Negatives),
    # ognuno delle dimensioni pari al numero di classi (meno uno, se si considera un indice da ignorare).
    # Questi tensori sono usati per accumulare conteggi globali su più batch di dati.
    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

        # Il metodo addBatch è un componente critico per calcolare le metriche di Intersection over Union (IoU) in compiti di segmentazione semantica.

    # Parametri di Input :
    # x: Predizioni del modello per un batch di immagini. Ha dimensioni [batch_size, nClasses, H, W], dove nClasses è il numero di classi di segmentazione, H e W sono rispettivamente altezza e larghezza dell'immagine.
    # y: Etichette di ground truth corrispondenti a x. Ha le stesse dimensioni di x.
    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be "batch_size x nClasses x H x W"

        # print ("X is cuda: ", x.is_cuda)
        # print ("Y is cuda: ", y.is_cuda)

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
            # y_onehot[:, self.ignoreIndex] estrae il canale dal tensore y_onehot che corrisponde all'ignoreIndex.
            # Questa operazione "taglia" attraverso la seconda dimensione del tensore (che rappresenta le classi) e seleziona solo la fetta corrispondente all'indice specificato.
            # Il risultato di questa operazione è un tensore di dimensione [batch_size, H, W], che rappresenta la presenza (o l'assenza) della classe specificata da self.ignoreIndex in ogni posizione dell'immagine per ogni immagine nel batch.
            # Ogni elemento in questo tensore risultante sarà 1 laddove la classe da ignorare è presente e 0 dove non lo è.
            # Nel tensore risultante, ogni pixel avrà un valore di 1 se appartiene alla classe ignoreIndex, indicando che quel pixel rappresenta la classe da ignorare.
            # Al contrario, i pixel avranno un valore di 0 se non appartengono a quella classe, indicando che non rappresentano la classe da ignorare.
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            # L'istruzione x_onehot = x_onehot[:, :self.ignoreIndex] in PyTorch è utilizzata per modificare il tensore x_onehot, che rappresenta le previsioni del modello in formato one-hot, escludendo la classe specificata da self.ignoreIndex.
            # Questa operazione è parte del processo di valutazione in compiti di segmentazione semantica, dove potrebbe essere necessario ignorare una classe specifica (come una classe di sfondo o di bordo)
            # x_onehot[:, :self.ignoreIndex] riduce il tensore x_onehot mantenendo tutte le classi tranne quella specificata dall'ignoreIndex.
            # La notazione [:, :self.ignoreIndex] significa:
            # :: seleziona tutti gli elementi lungo la dimensione del batch (prima dimensione).
            # :self.ignoreIndex: seleziona tutte le classi dalla prima fino alla classe immediatamente precedente a self.ignoreIndex. In altre parole, esclude la classe ignoreIndex e tutte quelle che la seguono.
            # Il risultato di questa operazione è un tensore che ha le stesse dimensioni di x_onehot ma con un numero ridotto di canali (classi), dove i canali che rappresentano la classe ignoreIndex e eventuali classi successive sono stati rimossi.
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        eps = 1e-8
        probs = torch.nn.functional.softmax(x_onehot, dim=1)
        entropy = torch.sum(-probs * torch.log(probs + eps), dim=1) / torch.log(torch.tensor(probs.shape[1]) + eps)
        normalized_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + eps)

        if x.is_cuda:
            entropy = entropy.cuda()

        x_onehot *= entropy.unsqueeze(1)

        # x_onehot e y_onehot sono tensori one-hot che rappresentano rispettivamente le previsioni del modello e le etichette di ground truth (verità di base). In un formato one-hot, ogni classe è rappresentata da un canale separato, e la presenza di una classe in una specifica posizione è indicata con un 1 in quel canale.
        # Questa operazione esegue una moltiplicazione elemento per elemento dei due tensori. Poiché entrambi sono in formato one-hot, il risultato (tpmult) avrà 1 nei punti in cui la previsione e il ground truth sono d'accordo (ovvero, entrambi hanno un 1 per la stessa classe nella stessa posizione) e 0 altrove. Questo identifica i True Positives.
        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        # Questa serie di funzioni sum somma i valori all'interno del tensore tpmult lungo varie dimensioni per ottenere il conteggio totale dei True Positives per ogni classe. Somme Cumulative Lungo Diverse Dimensioni:
        # 1 - torch.sum(tpmult, dim=0, keepdim=True): Calcola la somma lungo la dimensione del batch (dim=0). Dopo questa operazione, il tensore risultante non avrà più la dimensione del batch, ma manterrà le dimensioni originali grazie a keepdim=True.
        # 2 - torch.sum(..., dim=2, keepdim=True): Successivamente, calcola la somma lungo la dimensione dell'altezza dell'immagine (H).
        # 3-  torch.sum(..., dim=3, keepdim=True): Infine, calcola la somma lungo la dimensione della larghezza dell'immagine (W).
        # Il risultato finale, tp, è un tensore che contiene il conteggio totale dei True Positives per ogni classe sull'intero batch di immagini. Questo tensore è fondamentale per il calcolo dell'Intersection over Union (IoU) e altre metriche di valutazione nelle applicazioni di segmentazione semantica.
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        # Il prodotto di x_onehot e (1 - y_onehot - ignores) dà fpmult, che identifica i False Positives: i casi in cui il modello prevede una classe, ma il ground truth indica che quella posizione non appartiene a quella classe, escludendo le classi da ignorare.
        # fpmult = x_onehot * (1 - y_onehot - ignores) genera un tensore che ha valori 1 nei pixel dove il modello ha predetto erroneamente una classe (False Positive). Questo calcolo è fatto separatamente per ogni classe e per ogni pixel nell'immagine.
        fpmult = x_onehot * (
                    1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        # Ragionamento analogo a quello visto per i true positive. Importante segnalare che
        # sia fp,tp,fn sono tensori in cui ogni elemento rappresenta il conteggio totale dei False Positives (True Positive o False Negative) per una specifica classe sull'intero batch di immagini.
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        # L'istruzione fnmult = (1-x_onehot) * (y_onehot) nel contesto della segmentazione semantica in PyTorch è utilizzata per calcolare i False Negatives (FN). Ecco come funziona
        # 1-x_onehot: Calcola il complemento delle previsioni del modello. Per ogni classe e ogni pixel, dove c'è un 1 in x_onehot (indicando che il modello ha previsto quella classe), diventerà 0 nel complemento, e viceversa. Quindi, se x_onehot indica la presenza di una classe, 1-x_onehot indica dove il modello non ha rilevato quella classe.
        # * (y_onehot): Moltiplica il complemento delle previsioni (1-x_onehot) per le etichette di ground truth (y_onehot). Questo passaggio calcola dove il modello ha mancato di rilevare una classe che è effettivamente presente nel ground truth. In altre parole, identifica i False Negatives.
        # Dove y_onehot è 1 (la classe è presente nel ground truth) e x_onehot è 0 (il modello non ha rilevato la classe), il risultato sarà 1 (indicando un False Negative).
        # Dove y_onehot è 0 (la classe non è presente nel ground truth) o x_onehot è 1 (il modello ha rilevato la classe), il risultato sarà 0.
        # Il risultato di fnmult è un tensore che indica i False Negatives per ogni classe e per ogni pixel. In particolare, mostra dove il modello non ha rilevato una classe che è presente nel ground truth.
        fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        # Il numeratore per il calcolo dell'IoU è il numero di True Positives (TP) che sono stati accumulati usando il metodo addBatch. TP indica i casi in cui sia le previsioni del modello che le etichette di ground truth concordano sulla presenza di una classe.
        num = self.tp
        # Il denominatore è la somma di True Positives, False Positives (FP) e False Negatives (FN). FP rappresenta i casi in cui il modello prevede erroneamente una classe, mentre FN rappresenta i casi in cui il modello non rileva una classe presente nel ground truth. Il termine 1e-15 è aggiunto per evitare la divisione per zero.
        den = self.tp + self.fp + self.fn + 1e-15
        # Calcola l'IoU per ogni classe. L'IoU è una misura di quanto bene la previsione della classe si sovrappone con la sua etichetta di ground truth.
        iou = num / den
        # I valori ritornati dal metodo sono :
        # torch.mean(iou) --> Calcola la media dell'IoU su tutte le classi. Questa è la metrica Mean Intersection over Union (mIoU), che fornisce un singolo valore che rappresenta la performance media del modello su tutte le classi.
        # iou --> iou: Restituisce l'IoU per ogni classe individualmente. Questo permette di analizzare la performance del modello per ciascuna classe specifica
        return torch.mean(iou), iou  # returns "iou mean", "iou per class"


# Class for colors
class colors:
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


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

