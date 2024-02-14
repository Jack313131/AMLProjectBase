import torch
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
