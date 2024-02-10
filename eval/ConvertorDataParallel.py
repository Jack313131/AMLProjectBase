import torch
import torch.nn as nn
def convert_model_from_dataparallel(model):
    """
    Converte un modello PyTorch avvolto in DataParallel in un modello standard.

    Args:
    model (torch.nn.Module): Modello avvolto in DataParallel.

    Returns:
    torch.nn.Module: Modello non avvolto.
    """
    # Controlla se il modello è avvolto in DataParallel
    if isinstance(model, nn.DataParallel):
        # Estrai il modello interno
        model = model.module

    return model