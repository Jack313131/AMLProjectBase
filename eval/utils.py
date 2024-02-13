import contextlib
import copy
import sys
import subprocess
import torch.nn as nn
import thop
import torch
from pathlib import Path
import logging
import os
from erfnet import non_bottleneck_1d,DownsamplerBlock
from google.colab import drive

args = None
def run_command(command):
    print(f"Running the command :  {command} ...")
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print("Command execute correctly ...")
        return True, output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Command NOT executed correctly. Error : {e.output.decode('utf-8')}")
        return False, e.output.decode('utf-8')

def convert_model_from_dataparallel(model):

    if isinstance(model, nn.DataParallel):
        model = model.module

    return model

def connect_to_drive():
    path_drive = "/content/drive/MyDrive"
    if not os.path.exists(path_drive):
        print("Connecting to drive ... ")
        drive.mount('/content/drive')

    print("Drive connected ... ")

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
def compute_difference_flop(modelOriginal,modelPruning):

    logger = logging.getLogger('thop')
    logger.setLevel(logging.WARNING)

    input = torch.randn(1, 3, 512, 1024).to("cuda" if torch.cuda.is_available() else "cpu")
    with suppress_stdout():
        flopsOriginal, paramsOriginal = thop.profile(modelOriginal, inputs=(input,))
        flopsPruning, paramsPrunning = thop.profile(modelPruning, inputs=(input,))
    print(f"FLOPs modelOriginal : {flopsOriginal} - FLOPs modelPruning : {flopsPruning} the difference is : {flopsOriginal-flopsPruning}")
    print(f"Params modelOriginal : {paramsOriginal} - Params modelPruning : {paramsPrunning} the difference is : {paramsOriginal - paramsPrunning}\n")

    return flopsOriginal,flopsPruning,paramsOriginal,paramsPrunning

def remove_mask_from_model_with_pruning(model,state_dict):

    if isinstance(state_dict,nn.Module):
        return state_dict
    # Questa linea estrae lo stato attuale del modello, cioè i parametri attuali (pesi, bias, ecc.) del modello. state_dict() è una funzione PyTorch che restituisce un ordinato dizionario (OrderedDict) dei parametri.
    own_state = model.state_dict()
    # Il codice itera attraverso ogni coppia chiave-valore nel dizionario state_dict. name è il nome del parametro (per esempio, il nome di un particolare strato o peso nella rete), e param sono i valori effettivi dei pesi per quel nome.
    for name, param in state_dict.items():
        if "weight_orig" in name:
            # Il nome del parametro originale senza il suffisso '_orig'
            original_name = name.replace("_orig", "")
            # Recupera la maschera e il parametro originale
            mask = state_dict[name.replace("weight_orig", "weight_mask")]
            original_param = state_dict[name]
            # Applica la maschera al parametro
            pruned_param = original_param * mask
            name = original_name
            param = pruned_param
            # original_name = original_name.replace("module.","")
            # Aggiorna il parametro nel modello
            # getattr(model, original_name).data.copy_(pruned_param)
            # own_state[original_name].copy_(pruned_param)

        if "weight_mask" not in name:
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

    return remove_prunned_channels_from_model(model)

def print_and_save(output, file):
    print(output)
    file.write(output + "\n")

def save_model_mod_on_drive(model,filename):
    path_drive = args.path_drive
    Path(path_drive).mkdir(parents=True,exist_ok=True)
    dir_name = path_drive+filename.replace(".pth","")
    Path(dir_name).mkdir(parents=True,exist_ok=True)
    torch.save(model, path_drive+filename)
    print(f"The model {filename} has been saved on the path : {dir_name}")

def set_args(__args):
    args = __args
    args.add_argument("--path_drive", default="/content/drive/MyDrive/AML/")


def remove_prunned_channels_from_model(modelOriginal):
    modelOriginal = convert_model_from_dataparallel(modelOriginal)
    modelFinal = copy.deepcopy(modelOriginal)
    new_input_channel_next_layer = 0
    parent_module = None
    for nameLayer, layer in  modelOriginal.named_modules():
        if isinstance(layer, nn.Conv2d) and not 'output_conv' in nameLayer:
            if new_input_channel_next_layer > 0:
                in_channels = new_input_channel_next_layer
                input_changed = True
            else:
                in_channels = layer.in_channels
                input_changed = False
                non_zero_filters_prev_layer = None

            # Trova i filtri non azzerati (esempio ipotetico, la logica esatta può variare)
            non_zero_filters = layer.weight.data.sum(dim=(1, 2, 3)) != 0
            new_out_channels = non_zero_filters.long().sum().item()
            new_input_channel_next_layer = new_out_channels if new_out_channels != layer.out_channels else 0
            if new_out_channels != layer.out_channels or in_channels != layer.in_channels:

                if input_changed == False and layer.in_channels != new_out_channels:
                    new_layer_adapt_input = nn.Conv2d(layer.in_channels, new_out_channels, kernel_size=1, stride=1)
                    path_keys = str(nameLayer).split(".")
                    parent_module = modelFinal
                    for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                        parent_module = getattr(parent_module, key)

                    setattr(parent_module, "adaptingInput", new_layer_adapt_input)

                new_layer =  nn.Conv2d(in_channels=in_channels, out_channels=new_out_channels,
                      kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                      dilation=layer.dilation,bias=layer.bias is not None)

                if input_changed:
                    new_layer.weight.data = layer.weight.data[non_zero_filters][:,non_zero_filters_prev_layer,:,:]
                    if layer.bias is not None:
                        new_layer.bias.data = layer.bias.data[non_zero_filters]
                else:
                    new_layer.weight.data = layer.weight.data[non_zero_filters]
                    if layer.bias is not None:
                        new_layer.bias.data = layer.bias.data[non_zero_filters]


                path_keys = str(nameLayer).split(".")
                parent_module = modelFinal
                for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                    parent_module = getattr(parent_module, key)

                setattr(parent_module,path_keys[-1],new_layer)
                non_zero_filters_prev_layer = non_zero_filters

        if parent_module is not None and  isinstance(parent_module,DownsamplerBlock) and isinstance(layer, nn.Conv2d) and new_input_channel_next_layer == 0 :
            #new_input_channel_next_layer = new_layer.in_channels+new_layer.out_channels
            new_layer_adapt_input = nn.Conv2d(in_channels=new_layer.in_channels, out_channels=new_layer.out_channels, kernel_size=1, stride=1)
            setattr(parent_module, "adaptingInput", new_layer_adapt_input)

        if isinstance(layer, nn.BatchNorm2d) and  new_input_channel_next_layer > 0:
            new_bn = nn.BatchNorm2d(new_input_channel_next_layer, eps=layer.eps)
            path_keys = str(nameLayer).split(".")
            parent_module = modelFinal
            for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                parent_module = getattr(parent_module, key)
            setattr(parent_module,path_keys[-1],new_bn)

        if isinstance(layer,nn.ConvTranspose2d) and new_input_channel_next_layer >0 :
            new_layer = nn.ConvTranspose2d(in_channels=new_input_channel_next_layer, out_channels=layer.out_channels,
                                  kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                  output_padding = layer.output_padding,
                                  dilation=layer.dilation, bias=layer.bias is not None)
            path_keys = str(nameLayer).split(".")
            parent_module = modelFinal
            for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                parent_module = getattr(parent_module, key)
            setattr(parent_module, path_keys[-1], new_layer)
            new_input_channel_next_layer = 0


    return modelFinal
