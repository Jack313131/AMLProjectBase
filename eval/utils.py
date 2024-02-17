import contextlib
import copy
import shutil
import sys
import subprocess
import time
import re
import torch.nn as nn
import thop
import torch
from pathlib import Path
import logging
import os
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

from Loss import CrossEntropyLoss2d
from transform import ToLabel, Relabel
from dataset import cityscapes
from erfnet import non_bottleneck_1d,DownsamplerBlock
#from google.colab import drive
from torch.quantization import quantize_dynamic, prepare, convert
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize

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
        #drive.mount('/content/drive')

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
    modelOriginal = convert_model_from_dataparallel(modelOriginal).to("cuda" if torch.cuda.is_available() else "cpu")
    modelPruning = convert_model_from_dataparallel(modelPruning).to("cuda" if torch.cuda.is_available() else "cpu")
    with suppress_stdout():
        flopsOriginal, paramsOriginal = thop.profile(modelOriginal, inputs=(input,))
        flopsPruning, paramsPrunning = thop.profile(modelPruning, inputs=(input,))
    #print(f"FLOPs modelOriginal : {flopsOriginal} - FLOPs modelPruning : {flopsPruning} the difference is : {flopsOriginal-flopsPruning}")
    #print(f"Params modelOriginal : {paramsOriginal} - Params modelPruning : {paramsPrunning} the difference is : {paramsOriginal - paramsPrunning}\n")

    return flopsOriginal,flopsPruning,paramsOriginal,paramsPrunning

def remove_mask_from_model_with_pruning(model,state_dict):

    if isinstance(state_dict,nn.Module):
        print("Model passed is already a completed model ....")
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
                    own_state[name].copy_(param)
                    continue
            else:
                # se il modello partenza ha già un parametro con quel nome lo aggiorna direttamente
                # copy_ è una funzione in-place di PyTorch, il che significa che modifica direttamente il contenuto del tensore a cui si applica.
                # Poiché own_state[name] è un riferimento ai parametri reali del modello, questa operazione aggiorna direttamente i pesi all'interno del modello
                own_state[name].copy_(param)

    return remove_prunned_channels_from_model(model)
def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def print_and_save(output, file):
    print(output)
    file.write(remove_ansi_codes(output + "\n"))

def save_model_mod_on_drive(model,args):
    filename = args.modelFilenameDrive
    if not ".pth" in filename:
        filename = args.modelFilenameDrive+".pth"
    path_drive = args.path_drive+"ModelsExtra/"
    Path(path_drive).mkdir(parents=True,exist_ok=True)
    dir_name = path_drive+filename.replace(".pth","/")
    Path(dir_name).mkdir(parents=True,exist_ok=True)
    torch.save(model, dir_name+filename)
    print(f"The model {filename} has been saved on the path : {dir_name}")

def set_args(__args):
    args = __args
    features_model_input = None
    if hasattr(args,'loadWeightsPruned') and args.loadWeightsPruned is not None and 'model_best' in args.loadWeightsPruned and not 'erfnetPruningType' in args.loadWeightsPruned:
        features_model_input = args.loadWeightsPruned.replace("model_best","erfnet").replace("non_bottleneck","non bottleneck 1d").replace(".pth","").split("_")

        if features_model_input is not None and args.model is None:
            args.model = features_model_input[0]
        if features_model_input is not None and args.typePruning is None:
            args.typePruning = features_model_input[2]
        if features_model_input is not None and args.pruning is None:
            args.pruning = float(int(features_model_input[3])/10)
        if features_model_input is not None and args.typeNorm is None:
            args.typeNorm = int(features_model_input[4].replace("L",""))
        if features_model_input is not None and args.listLayerPruning is None:
            args.listLayerPruning = [nameModule.replace(" ","_") for nameModule in features_model_input[features_model_input.index('module')+1 : features_model_input.index('Layer')]]
        if features_model_input is not None and args.listNumLayerPruning is None and features_model_input[features_model_input.index('Layer')+1] !="All":
            args.listNumLayerPruning = [int(numLayer) for numLayer in [features_model_input[features_model_input.index('Layer')+1:]]]

    condition1 = hasattr(args,
                         'loadWeightsPruned') and args.loadWeightsPruned is not None and 'erfnetPruningType' in args.loadWeightsPruned
    condition2 = hasattr(args,'loadModelPruned') and args.loadModelPruned is not None and 'erfnetPruningType' in args.loadModelPruned
    if condition1 or condition2:


        if condition1:
            features_model_input = args.loadWeightsPruned.split("/")[-1]
        elif condition2:
          features_model_input = args.loadModelPruned.replace('modelPrunnedCompleted/',"").replace("(_conv_bn)","").split("/")[-1]


        features_model_input = features_model_input.replace("erfnetPruningType", "erfnet").replace("model_best_","").replace("non_bottleneck_1d","non bottleneck 1d").replace("(_conv)","").replace(".pth", "").split("_")

        if features_model_input is not None and not hasattr(args,'model'):
            args.model = features_model_input[0]
        if features_model_input is not None and (not hasattr(args,'typePruning') or args.typePruning is None):
            args.typePruning = features_model_input[1]
        if features_model_input is not None and (not hasattr(args,'pruning') or args.pruning is None):
            args.pruning = float(features_model_input[5])
        if features_model_input is not None and (not hasattr(args,'typeNorm') or args.typeNorm is None):
            args.typeNorm = int(features_model_input[3])
        if features_model_input is not None and (not hasattr(args,'listLayerPruning') or len(args.listLayerPruning)==0):
            args.listLayerPruning = [nameModule.replace(" ", "_") for nameModule in features_model_input[features_model_input.index('Layer') + 1: features_model_input.index('NumLayerPruning')]]
        if features_model_input is not None and (not hasattr(args,'listNumLayerPruning') or features_model_input[features_model_input.index('Layer') + 1] != "All"):
            args.listNumLayerPruning = [int(numLayer) for numLayer in features_model_input[features_model_input.index('NumLayerPruning') + 1:]]

    args.path_drive="/content/drive/MyDrive/AML/"
    args.modelFilenameDrive =define_name_model(args)
    args.savedir = args.modelFilenameDrive

    return args

def remove_prunned_channels_from_model(modelOriginal):
    modelOriginal = convert_model_from_dataparallel(modelOriginal)
    modelFinal = copy.deepcopy(modelOriginal)
    new_input_channel_next_layer = 0
    parent_module = None
    for nameLayer, layer in modelOriginal.named_modules():
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

                if new_out_channels != in_channels:
                    new_layer_adapt_input = nn.Conv2d(in_channels=in_channels, out_channels=new_out_channels, kernel_size=1, stride=1)
                    path_keys = str(nameLayer).split(".")
                    parent_module = modelFinal
                    for key in path_keys[0:-1]:  # Vai fino al genitore del layer
                        parent_module = getattr(parent_module, key)

                    if not hasattr(parent_module,"adaptingInput"):
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

def quantize_model_to_int8(model,input_transform_cityscapes,target_transform_cityscapes,args):
    print("Preparing the model for the quantization to int8")
    model.eval()
    model_fp32_prepared = prepare(model)
    model_fp32_prepared.eval()
    calibration_dataloader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset='val'),
        num_workers=torch.cuda.device_count(), batch_size=6, shuffle=False)
    print("Starting calibration ... ")
    for batch in calibration_dataloader:
        model_fp32_prepared(batch)

    print("Staring conversation model to int8 ...")
    model_int8 = convert(model_fp32_prepared)

    print("Saving the model ...")
    torch.save(model_int8, f"{args.path_drive}/Models/Model_int8/{args.modelFilenameDrive}_model_int8.pth")

    return model_int8

def define_name_model(args):
    if args.pruning > 0:
        typeNorm = f"_Norm_{args.typeNorm}" if args.typeNorm else ""
        namePruning = f"PruningType_{args.typePruning}{typeNorm}_Value_{args.pruning}"
        if len(args.moduleErfnetPruning) > 0:
            namePruning = namePruning + "_Module"
            for module in args.moduleErfnetPruning:
                namePruning = namePruning + f"_{module}"
        if len(args.listLayerPruning) > 0:
            namePruning = namePruning + "_Layer"
            nameInnerStateMod = "("
            for value in args.listInnerLayerPruning:
                nameInnerStateMod = nameInnerStateMod + f"_{value}"
            nameInnerStateMod += ")"
            for layer in args.listLayerPruning:
                namePruning = namePruning + f"_{layer}{nameInnerStateMod}"
            numberLayer = "_AllLayer"
            if len(args.listNumLayerPruning) > 0:
                numberLayer = "_NumLayerPruning"
                for number in args.listNumLayerPruning:
                    numberLayer = numberLayer + f"_{number}"

            namePruning = namePruning + numberLayer

    return args.model + ("FreezingBackbone" if args.freezingBackbone else "") + (
        namePruning if args.pruning else "")

def training_new_layer_adapting(model,input_transform_cityscapes,target_transform_cityscapes,weight,args):
    dataset_extra = args.datadir+"_extra"
    print(f"Using the dataset {dataset_extra}")
    loader_finetuning_adapting_layers = DataLoader(
        cityscapes(dataset_extra, input_transform_cityscapes, target_transform_cityscapes, subset='train'),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # commando per forzare il modello ad essere in modelità valutazione
    model.train()

    if torch.cuda.is_available():
        model = model.to("cuda")
        weight = weight.cuda()

    criterion = CrossEntropyLoss2d(weight)
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    max_lr = 0.01  # Il massimo learning rate
    num_epochs = 20
    steps_per_epoch = len(loader_finetuning_adapting_layers)  # Numero di batch (iterazioni) per epoca
    total_steps = num_epochs * steps_per_epoch  # Numero totale di iterazioni

    # Inizializzazione dello scheduler
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    print("Start fine tuning pruning for adding layers ... ")
    print("Freezing layer not named : adaptingInput")
    for name, param in model.named_parameters():
        # Congela i parametri se 'adaptingInput' non è nel nome del layer
        if 'adaptingInput' not in name:
            param.requires_grad = False
        else:
            # Assicurati che i parametri che non devono essere congelati siano settati per il gradiente
            param.requires_grad = True

    for epoch in range(1, num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        epoch_loss = []
        time_train = []
        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        for step, (images, labels, _, _) in enumerate(loader_finetuning_adapting_layers):

            start_time = time.time()

            # Prima di calcolare i gradienti per l'epoca corrente, è necessario azzerare i gradienti accumulati dalla bacth precedente.
            # Questo è essenziale perché, per impostazione predefinita, i gradienti si sommano in PyTorch per consentire l'accumulo di gradienti in più passaggi.
            optimizer.zero_grad()

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            images.requires_grad_(True)
            inputs = images

            # labels.requires_grad_(True)
            targets = labels

            outputs = model(inputs)
            # print("Outputs: ", outputsOriginal.shape)

            loss = criterion(outputs, targets[:, 0])
            optimizer.zero_grad()
            # Questo calcola i gradienti della perdita rispetto ai parametri del modello. È il passo in cui il modello "impara", aggiornando i gradienti in modo da minimizzare la perdita.
            loss.backward()

            optimizer.step()

            scheduler.step()  ## scheduler 2
            # epoch_loss è un vettore in cui sono aggiunti ad ogni batch il valore ritornato dalla loss function
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
            if step % 50 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

    print("Model Pruned Completely ... ")
    save_model_mod_on_drive(model=model,args = args)
    print(f"Model Pruned Completely has been saved on the path {args.path_drive}ModelsExtra/{args.modelFilenameDrive}/")
def saveOnDrive(epoch=None, model="", pathOriginal="",args=None):
    pathOriginal = f"/content/AMLProjectBase/save/{args.savedir}/"
    model = args.modelFilenameDrive
    if not os.path.isdir(pathOriginal):
        print(f"Path Original is wrong : {pathOriginal}")
    drive = "/content/drive/MyDrive/"
    if os.path.isdir(drive):
        if not os.path.isdir(drive + f"AML/"):
            os.mkdir(drive + f"AML/")
        if not os.path.exists(drive + f"AML/{model}/"):
            os.mkdir(drive + f"AML/{model}/")
        if os.path.exists(pathOriginal + "/checkpoint.pth.tar"):
            shutil.copy2(pathOriginal + "/checkpoint.pth.tar", drive + f"AML/{model}/checkpoint.pth.tar")
        if os.path.exists(pathOriginal + "/automated_log.txt"):
            shutil.copy2(pathOriginal + "/automated_log.txt", drive + f"AML/{model}/automated_log.txt")
        if os.path.exists(pathOriginal + "/opts.txt"):
            shutil.copy2(pathOriginal + "/opts.txt", drive + f"AML/{model}/opts.txt")
        if os.path.exists(pathOriginal + "/model.txt"):
            shutil.copy2(pathOriginal + "/model.txt", drive + f"AML/{model}/model.txt")
        if os.path.exists(pathOriginal + "/pruning_setting.txt"):
            shutil.copy2(pathOriginal + "/pruning_setting.txt", drive + f"AML/{model}/pruning_setting.txt")
        if os.path.isfile(pathOriginal + "/model_best.pth"):
            shutil.copy2(pathOriginal + "/model_best.pth", drive + f"AML/{model}/model_best.pth")
        if os.path.isfile(pathOriginal+ f"/model_best_{args.modelFilenameDrive}.pth"):
            shutil.copy2(pathOriginal+ f"/model_best_{args.modelFilenameDrive}.pth", drive + f"AML/{model}/model_best_{args.modelFilenameDrive}.pth")
        if os.path.isfile(pathOriginal + "/result.txt"):
            shutil.copy2(pathOriginal + "/result.txt", drive + f"AML/{model}/result.txt")
        if epoch is not None:
            print(f"Checkpoint of epoch {epoch} saved on Drive path : {drive}AML/{model}/")
        if epoch is None:
            path_drive = f"{drive}AML/{model}/"
            print(f"Saved on drive on the path {path_drive}")
    else:
        print("Drive is not linked ...")

def direct_quantize(args, model, test_loader):
    for i, (data, target, filename, filenameGt) in enumerate(test_loader, 1):
        if not args.cpu:
            data = data.cuda()
            target = target.cuda()
        model.quantize_forward(data)
        if args.cpu and i % 10 == 0:
            break
        if i % 500 == 0:
            break
    print('direct quantization finished')