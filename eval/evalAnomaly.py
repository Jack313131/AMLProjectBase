# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

# Un "seed" è un valore iniziale utilizzato per inizializzare un generatore di numeri casuali. I generatori di numeri casuali sono algoritmi che producono una sequenza di numeri che non mostrano alcun pattern prevedibile.
# Tuttavia, se inizializzi un generatore di numeri casuali con un valore di seed specifico, esso produrrà sempre la stessa sequenza di numeri ogni volta che viene eseguito.
# Dato che molti algoritmi di machine learning (come l'inizializzazione dei pesi di una rete neurale, la divisione dei dati in set di training e test, etc.) dipendono da operazioni casuali, utilizzare un seed fisso assicura che queste operazioni siano consistenti tra diverse esecuzioni.
# Confronto tra Modelli: Quando si sperimentano diverse architetture di rete o si regolano i parametri, è importante che ogni variante sia testata nelle stesse condizioni. Utilizzando lo stesso seed, puoi assicurarti che ogni modello parta con le stesse inizializzazioni e selezioni gli stessi dati, permettendo un confronto equo.
seed = 42

# Impostando lo stesso valore di seed e utilizzando le istruzioni random.seed(seed), np.random.seed(seed), e torch.manual_seed(seed), assicuri che ogni volta che esegui il codice,
# le funzioni di generazione di numeri casuali in Python (modulo random), NumPy, e PyTorch produrranno la stessa sequenza di numeri casuali, rispettivamente
# Ogni libreria (Python random, NumPy, PyTorch) utilizza il proprio generatore di numeri casuali. Impostare lo stesso seed in ciascuna di queste librerie non significa che produrranno la stessa sequenza l'una rispetto all'altra. Significa semplicemente che la sequenza prodotta da ogni libreria sarà la stessa ogni volta che esegui lo script.

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# NUM_CHANNELS = 3 indica che le immagini da elaborare hanno 3 canali (tipicamente RGB - Rosso, Verde, Blu).
# NUM_CLASSES = 20 specifica che il modello si occupa di classificare i pixel in una delle 20 classi diverse. Questo è tipico in compiti di segmentazione semantica dove ogni classe rappresenta una categoria diversa (come strade, veicoli, pedoni, ecc.).
NUM_CHANNELS = 3
NUM_CLASSES = 20

# Queste righe configurano il modo in cui PyTorch interagisce con CUDA Deep Neural Network library (cuDNN), un'acceleratore di GPU fornito da NVIDIA.

# Questa impostazione fa sì che l'algoritmo scelto da cuDNN per le operazioni convoluzionali sia deterministico, cioè produca risultati consistenti e riproducibili. È utile per la riproducibilità, ma può ridurre l'efficienza in termini di performance.
torch.backends.cudnn.deterministic = True
# Quando attivato, permette a cuDNN di eseguire automaticamente benchmark e selezionare l'algoritmo più efficiente per le operazioni convoluzionali in base alle dimensioni della rete e delle immagini di input. Questo può migliorare le prestazioni durante l'addestramento, ma può portare a risultati leggermente diversi ad ogni esecuzione.
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # La funzione load_my_state_dict nel codice che hai fornito è progettata per caricare i pesi (o parametri) di un modello di machine learning da un file salvato
    # La funzione accetta due argomenti: model, che è il modello di machine learning in cui si desidera caricare i pesi, e state_dict, che è un dizionario contenente i pesi da caricare.
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        # Questa linea estrae lo stato attuale del modello, cioè i parametri attuali (pesi, bias, ecc.) del modello. state_dict() è una funzione PyTorch che restituisce un ordinato dizionario (OrderedDict) dei parametri.
        own_state = model.state_dict()
        # Il codice itera attraverso ogni coppia chiave-valore nel dizionario state_dict. name è il nome del parametro (per esempio, il nome di un particolare strato o peso nella rete), e param sono i valori effettivi dei pesi per quel nome.
        for name, param in state_dict.items():
            if name not in own_state:
                # Se il nome inizia con "module.", questo suggerisce che il dizionario dei pesi proviene da un modello che è stato addestrato usando DataParallel,
                # che aggiunge il prefisso "module." a tutti i nomi dei parametri. In questo caso, il codice cerca di adattare i nomi dei parametri rimuovendo "module." e tenta nuovamente di caricare il peso nel modello.
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param) # name.split("module.")[-1] --> toglie la parte module. dal nome e si tiene il resto
                else:
                    print(name, " not loaded")
                    continue
            else:
                # se il modello partenza ha già un parametro con quel nome lo aggiorna direttamente
                # copy_ è una funzione in-place di PyTorch, il che significa che modifica direttamente il contenuto del tensore a cui si applica.
                # Poiché own_state[name] è un riferimento ai parametri reali del modello, questa operazione aggiorna direttamente i pesi all'interno del modello
                own_state[name].copy_(param)
        return model # l'aggiornamento diretto dei pesi viene fatto copiando i nuovi valori dei pesi nei opportuni layer del own_state il quale tiene un puntatore dei vari layer (cosi la modifica si propaga subito anche al modello di partenza)

    # torch.load è una funzione in PyTorch che carica un oggetto salvato da un file. Questo oggetto può essere qualsiasi cosa che sia stata salvata precedentemente con torch.save, come un modello, un dizionario di stato del modello, un tensore, ecc.
    # map_location è un argomento di torch.load che specifica come e dove i tensori salvati devono essere mappati in memoria. Può essere utilizzato per forzare tutti i tensori ad essere caricati su CPU o su una specifica GPU, o per mapparli da una configurazione di hardware a un'altra.
    # Nel tuo esempio, map_location=lambda storage, loc: storage è una funzione lambda che ignora il loc (la localizzazione originale del tensore quando è stato salvato) e restituisce storage. Questo significa che i tensori saranno caricati sulla stessa tipologia di dispositivo da cui sono stati salvati (CPU o GPU).
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    # commando per forzare il modello ad essere in modelità valutazione
    model.eval()

    # os.path.expanduser espande eventuali riferimenti alla home directory dell'utente nel percorso. Ad esempio, se il percorso inizia con ~, viene sostituito con il percorso della directory home dell'utente. Per esempio, ~/Documents potrebbe essere espanso in /home/username/Documents su un sistema Linux.
    #glob è un modulo Python che trova tutti i percorsi di file che corrispondono a un pattern specificato. glob.glob prende un pattern di percorso di file e restituisce un elenco di percorsi di file che corrispondono a quel pattern. (in questo caso quello specificato nel input)
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        # Questa linea carica un'immagine dal percorso path, la converte in RGB (nel caso non lo sia già), la trasforma in un array NumPy, poi in un tensore PyTorch, e infine aggiunge una dimensione iniziale (questo è fatto per simulare un batch di dimensione 1, come richiesto dai modelli PyTorch)
        # 1 ) Image.open(path): Apre l'immagine dal percorso specificato in path. Image è tipicamente dal modulo PIL o Pillow, una libreria di manipolazione delle immagini in Python.
        # 2 ) .convert('RGB'): Converte l'immagine nel formato RGB (Rosso, Verde, Blu). Questo è importante per assicurarsi che l'immagine sia in un formato standard a tre canali, specialmente se l'immagine originale è in scala di grigi o ha un canale alfa (trasparenza).
        # 3 ) np.array(...) Trasforma l'immagine PIL in un array NumPy. Questo è un passo comune per interfacciare le immagini con le librerie di calcolo scientifico come NumPy.
        # 4 ) torch.from_numpy(...) Converte l'array NumPy in un tensore PyTorch. PyTorch è una libreria di deep learning che utilizza tensori per le operazioni di calcolo.
        # 5 ) Aggiunge una dimensione extra all'inizio del tensore. Questo trasforma il tensore da [altezza, larghezza, canali] a [1, altezza, larghezza, canali]. In PyTorch, questa dimensione extra rappresenta il batch size. Anche se stai lavorando con una singola immagine, i modelli generalmente si aspettano un batch di immagini come input.
        #     Più specificamente, .unsqueeze(0) aggiunge una nuova dimensione all'indice 0 del tensore. Questo significa che se avevi un tensore con una certa forma (o dimensione), dopo aver chiamato .unsqueeze(0), la forma del tensore avrà una dimensione in più all'inizio. I modelli di deep learning in PyTorch di solito aspettano che i dati siano forniti in un batch. Anche se stai lavorando con una singola immagine (o un singolo dato), devi ancora fornirla come se fosse un batch di dimensione 1. .unsqueeze(0) aggiunge efficacemente la dimensione del batch, facendo sembrare che stai passando un batch di una singola immagine, anziché una singola immagine.
        # 6 ) .float(): Converte i dati del tensore in un formato a virgola mobile. Questo è necessario perché molti modelli PyTorch lavorano con float piuttosto che con interi.
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()

        # Modifica l'ordine delle dimensioni del tensore. In PyTorch, l'ordine standard dei dati dell'immagine è [batch size, canali, altezza, larghezza] (NCHW). Tuttavia, le immagini caricate con PIL e trasformate in array NumPy hanno un formato [batch size, altezza, larghezza, canali] (NHWC). Quindi, .permute(0,3,1,2) riordina le dimensioni per adattarle all'ordine NCHW atteso dai modelli PyTorch.
        images = images.permute(0,3,1,2)
        # Il modello viene eseguito sulle immagini senza calcolare i gradienti (perché è in fase di valutazione, non di addestramento).
        with torch.no_grad():
            result = model(images)

        # L'istruzione sotto non effettua un confronto con il ground truth, ma piuttosto valuta quanto il modello è sicuro della sua predizione. Questo calcolo fornisce un punteggio di anomalia basato sulla confidenza della previsione del modello per ciascun pixel, ma non si confronta direttamente con le etichette reali o il ground truth.
        # Questo calcolo è indipendente dal ground truth. Non sta valutando l'accuratezza o la precisione del modello rispetto alle etichette reali. Piuttosto, sta valutando quanto il modello è "sicuro" nelle sue previsioni, indipendentemente dal fatto che queste siano corrette o meno.
        # Sottraendo questo valore massimo da 1.0, si ottiene un punteggio che rappresenta l'incertezza o l'anomalia. Un punteggio vicino a 0 indica alta confidenza (bassa anomalia), mentre un punteggio vicino a 1 indica bassa confidenza (alta anomalia).
        # Il punteggio di anomalia risultante è sempre nel range tra 0 e 1. Un valore più vicino a 1 indica che il modello ha trovato qualcosa di insolito o meno prevedibile nell'immagine, che potrebbe essere un'indicazione di anomalia.
        # Infatti se il modello trovo un oggetto per cui si allenato nella segmentazione (uno delle 20 classi passate nella fase di training) il valore finale del istruzione sarà vicino a 0, poichè il valore massimo trovato per una classe di classificazione sarà vicino a 1. Al contrario se lo score della classificazione più probabile ha un valore basso, il valore del anomalia sarà vicino ad 1
        # I vari commandi nel istruzione significano :
        # 1 ) .squeeze(0) rimuove la dimensione di indice 0 dal tensore, a condizione che abbia dimensione 1 (tipicamente la dimensione del batch). Se result ha una forma di [1, C, H, W], dopo .squeeze(0), la sua forma diventa [C, H, W]. Questo passaggio è necessario per rimuovere la dimensione del batch, lasciando solo le dimensioni dell'output del modello.
        #     come risulato del output il modello ritorna un vettore [BS,C,W,H] dove il primo elemento rappresenta la batch size (1 in questo caso perchè consideriamo un immagine alla volta), C rappresenta il numero totale delle classi che il modello riesce a distinguere, H e W le info spaziali del pixel. Infatti dopo lo squeeze(0) che rimuve la dimensione del batch ottengo un vettore [C,W,H] ovvero per ogni pixel del immagine pixel(h,w) ho uno sccore per ogni C classe (il modello ha 20 classi)
        # 2a ) .data accede ai dati del tensore.
        # 2b ) .cpu() sposta il tensore sulla CPU (se era su una GPU).
        # 2c ) .numpy() converte il tensore PyTorch in un array NumPy. Questo è spesso fatto per l'elaborazione post-processing o per l'analisi, poiché NumPy offre una vasta gamma di operazioni e funzioni comode.
        # 3 ) np.max(..., axis=0): Calcola il valore massimo lungo l'asse specificato. axis =0 indica che il massimo viene calcolato lungo il primo asse (che, dopo il .squeeze(0), rappresenta il primo asse dell'output del modello). In pratica, questo significa che per ogni posizione [H, W] nel risultato, viene preso il valore massimo tra tutte le classi previste [C] (ovvero per ogni pixel viene preso la classe più probabile)
        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)

        # Viene calcolato il path del immagine di ground truth
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")

        # Apre l'immagine della maschera di ground truth dal percorso specificato pathGT. Questa immagine rappresenta le etichette reali per ciascun pixel dell'immagine corrispondente.
        mask = Image.open(pathGT)
        # Converte l'immagine PIL in un array NumPy. Ora, ood_gts contiene le etichette di ground truth come array.
        # Se l'immagine della ground truth (pathGT) è in scala di grigi e la converti in un array NumPy usando np.array(mask), la variabile ood_gts conterrà effettivamente una rappresentazione dell'immagine ground truth, ma non come un vettore lineare. Invece, sarà un array bidimensionale dove ogni elemento rappresenta un pixel dell'immagine ground truth.
        # In una immagine in scala di grigi, ogni pixel può essere rappresentato da un singolo valore di intensità (di solito da 0 a 255, dove 0 è nero e 255 è bianco).
        # Quando converti questa immagine in un array NumPy, ottieni un array 2D (due dimensioni) dove le dimensioni corrispondono alle dimensioni dell'immagine (altezza e larghezza).
        # Ogni elemento in questo array 2D è un numero che rappresenta l'intensità di un pixel nell'immagine.
        # Ad esempio, se l'immagine ha dimensioni 100x100 pixel, l'array ood_gts sarà di forma 100x100, con 10.000 elementi in totale, ciascuno corrispondente a un pixel dell'immagine.
        ood_gts = np.array(mask)

        # Adattamento delle Etichette in Base al Dataset: Le seguenti condizioni e trasformazioni sono specifiche per diversi dataset, presumibilmente utilizzati per il rilevamento di anomalie stradali.
        # Ogni blocco if verifica se il percorso della maschera di ground truth appartiene a un particolare dataset e poi applica trasformazioni specifiche a quell'insieme di dati.
        # Le etichette con valore x nel ground truth potrebbe non corrispondere con l'etichetta con cui il modello è stato allenato. Per cui le varie etichetture con numero discorde vengono trasformate all'etichettature del modello.
        if "RoadAnomaly" in pathGT:
            # Se il dataset è "RoadAnomaly", le etichette con valore 2 vengono trasformate in 1 (1 per ERFNet rappresenta side-walks)
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            # Se il dataset è "LostAndFound", le etichette con valore 0 vengono trasformate in 255 (255 per ERFNet rappresenta background)
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            # Se il dataset è "LostAndFound", le etichette con valore 1 vengono trasformate in 0 (0 per ERFNet rappresenta road)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            # Se il dataset è "LostAndFound", le etichette con valore tra 1 escluso e 201 escluso vengono trasformate in 1 (255 per ERFNet rappresenta side-walk)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            # Se il dataset è "Streethazard", le etichette con valore 14 vengono trasformate in 255 (255 per ERFNet rappresenta background)
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            # Se il dataset è "Streethazard", le etichette con valore minore di 20 vengono trasformate in 0 (0 per ERFNet rappresenta road)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            # Se il dataset è "Streethazard", le etichette con valore 255 vengono trasformate in 1 (1 per ERFNet rappresenta side-walks)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        # Questa istruzione controlla se il valore 1 è presente nell'array delle etichette di ground truth. Il valore 1 una classe specifica (side walk ?), probabilmente associata a una certa tipologia di anomalia.
        # np.unique(ood_gts): Restituisce gli elementi unici nell'array ood_gts, che contiene le etichette di ground truth.
        # se i valore 1 NON è presente allora continua a ciclare
        if 1 not in np.unique(ood_gts):
            continue
        else:
             # Se il valore 1 è presente in ood_gts, il blocco else viene eseguito.
             # ood_gts_list.append(ood_gts): Aggiunge le etichette di ground truth (ood_gts) alla lista ood_gts_list. Questo viene fatto per tenere traccia delle etichette di ground truth delle immagini che contengono anomalie.
             ood_gts_list.append(ood_gts)
             # anomaly_score_list.append(anomaly_result): Aggiunge il punteggio di anomalia calcolato (anomaly_result) alla lista anomaly_score_list. Questo permette di tenere traccia dei punteggi di anomalia associati a ciascuna immagine.
             anomaly_score_list.append(anomaly_result)
        # Questa istruzione elimina alcune variabili per liberare la memoria. È una buona pratica per la gestione delle risorse, soprattutto quando si lavora con un grande numero di immagini o con modelli che utilizzano molta memoria.
        del result, anomaly_result, ood_gts, mask
        #Questo comando libera la cache non utilizzata della GPU. È utile in scenari di deep learning per gestire in modo efficiente la memoria della GPU, specialmente quando si lavora con modelli di grandi dimensioni o con grandi set di dati.
        torch.cuda.empty_cache()

    file.write( "\n")

    #ood_gts e anomaly_scores vengono creati come array NumPy dalle liste ood_gts_list e anomaly_score_list.
    # ood_gts contiene le etichette di ground truth, mentre anomaly_scores contiene i punteggi di anomalia calcolati per ogni immagine.
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # ood_mask è una maschera booleana che identifica dove le etichette di ground truth sono uguali a 1 (presumibilmente indicando anomalie).
    ood_mask = (ood_gts == 1)
    # ind_mask è una maschera booleana per le etichette di ground truth uguali a 0 (presumibilmente indicando non-anomalie o dati in-distribution).
    ind_mask = (ood_gts == 0)

    # ood_out contiene i punteggi di anomalia corrispondenti alle anomalie rilevate (dove ood_gts è 1).
    ood_out = anomaly_scores[ood_mask]
    # ind_out contiene i punteggi di anomalia per i dati non-anomali (dove ood_gts è 0).
    ind_out = anomaly_scores[ind_mask]

    # ood_label è un array di 1, che rappresenta le etichette per i dati anomali.
    # ind_label è un array di 0, che rappresenta le etichette per i dati non-anomali.
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    # val_out contiene tutti i punteggi di anomalia (sia per dati anomali che non).
    # val_label contiene tutte le etichette corrispondenti (0 per non-anomali, 1 per anomali).
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    # prc_auc calcola l'AUC (Area Under the Curve) per la Precision-Recall Curve, una metrica comune per valutare la performance dei modelli in compiti di classificazione.
    prc_auc = average_precision_score(val_label, val_out)
    # fpr calcola il False Positive Rate al 95% di True Positive Rate, una metrica specifica per valutare la capacità del modello di rilevare correttamente le anomalie mantenendo basso il tasso di falsi positivi.
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()