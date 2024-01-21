import numpy as np

from torch.autograd import Variable

from visdom import Visdom

# La classe Dashboard è progettata per fornire funzionalità di visualizzazione durante l'addestramento di modelli di deep learning, utilizzando la libreria Visdom.
# Visdom è uno strumento per creare, organizzare e condividere visualizzazioni di dati in tempo reale.
# La classe Dashboard ha metodi per visualizzare perdite e immagini.
class Dashboard:

    def __init__(self, port):
        # Questa linea inizializza un'istanza di Visdom collegandosi a un server Visdom sulla porta specificata.
        # Visdom permette la visualizzazione di dati in tempo reale attraverso un'interfaccia web. Visdom funziona come un'applicazione server-cliente
        # Il server Visdom deve essere in esecuzione in ascolto su una specifica porta del computer. Quando crei un'istanza di Visdom nel tuo codice Python, devi specificare la porta su cui il server Visdom è in esecuzione, in modo che il tuo codice (il cliente) possa connettersi correttamente ad esso.
        self.vis = Visdom(port=port)

    # Questo metodo è progettato per visualizzare la perdita (loss function) durante l'addestramento. Riceve come input
    # 1. losses --> array di loss dove ogni elemento potrebbe contenere un epoch o batch (dipende come aggiunti elementi al array loss)
    # 2. title  --> titolo della visualizzazione
    def loss(self, losses, title):
        # permette di creare un array che parte da 1 ha lunghezza losses.len +1 ed ogni step si incrementa di 1. Questo ci permette di definire
        # asse x di lunghezza pari al numero di loss messe nel array
        x = np.arange(1, len(losses)+1, 1)

        # vis.line permette di disegnare un grafico a linee delle perdite
        # i primi due parametri corrispondono rispettivamente ad Y (losses) e x. Il parametro env specifica l'ambiente Visdom in cui visualizzare il grafico, mentre opts permette di personalizzare il grafico (ad esempio, impostando il titolo).
        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    # Questo metodo è progettato per visualizzare immagini
    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        # Se image è un oggetto Variable, image.data estrae il tensore sottostante da Variable.. Prima della deprecazione di Variable, questa era la prassi comune per ottenere il tensore originale senza il suo involucro di Variable.
        # Prima dell'introduzione di PyTorch 0.4, Variable e Tensor erano due classi distinte. Variable era usato per calcoli automatici e gradienti, mentre Tensor era una struttura dati più semplice.
        # Dalla versione 0.4, le funzionalità di Variable sono state integrate direttamente in Tensor, rendendo l'uso di Variable obsoleto. Da allora, tutte le operazioni che potevano essere eseguite su Variable possono essere eseguite direttamente su Tensor.
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image, env='images', opts=dict(title=title))