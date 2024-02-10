import torch
import torchvision.models as models
import torch.quantization
from train.erfnet import Net
from train.main import MyCoTransform
import torch
import torch.quantization
from train.dataset import VOC12, cityscapes
from torch.utils.data import DataLoader, Subset

transform = MyCoTransform(False, augment=False, height=512)

# Assumi che 'MyDataset' sia il tuo set di dati personalizzato
dataset = cityscapes("../dataset/datasetCityscapes", transform, 'train')

# Utilizza un subset del dataset per la calibrazione
subset_size = 100  # Numero di esempi da utilizzare per la calibrazione
subset_indices = torch.randint(0, len(dataset), (subset_size,))
calibration_dataset = Subset(dataset, subset_indices)

# Definisci il DataLoader
calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=6,  # Sostituisci con la dimensione del batch desiderata
    shuffle=True,
    num_workers=torch.cuda.device_count(),  # Sostituisci con il numero di worker desiderati
)

# Carica il modello pre-addestrato
model = Net(20)
model.load_state_dict(torch.load('../trained_models/erfnet_pretrained.pth'))
model.eval()

# Definisci la configurazione di quantizzazione
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepara il modello per la quantizzazione
model_prepared = torch.quantization.prepare(model)

# Calibra il modello
with torch.no_grad():
    for input, _ in calibration_dataset:
        model_prepared(input)

# Converti il modello in quantizzato
model_quantized = torch.quantization.convert(model_prepared)

# Salva o testa il modello quantizzato
model_quantized.save('model_quantized.pth')
# Test...
