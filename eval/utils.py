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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def is_drive_connect():
    return os.path.exists("/content/drive/MyDrive")

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
        return state_dict,False
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

    return remove_prunned_channels_from_model(model),True
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
    path_drive = args.path_drive+f"ModelsExtra/{args.load_dir_model_mod}/"
    Path(path_drive).mkdir(parents=True,exist_ok=True)
    dir_name = path_drive+filename.replace(".pth","/")
    Path(dir_name).mkdir(parents=True,exist_ok=True)
    torch.save(model, dir_name+filename)
    print(f"The model {filename} has been saved on the path : {dir_name}")

def save_checkpoint_mod_on_drive(checkpoint,args,epoch):
    filename = args.modelFilenameDrive
    if not ".pth" in filename:
        filename = args.modelFilenameDrive+".pth"
    path_drive = args.path_drive+f"ModelsExtra/{args.load_dir_model_mod}/"
    Path(path_drive).mkdir(parents=True,exist_ok=True)
    dir_name = path_drive+filename.replace(".pth","/")
    Path(dir_name).mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint, dir_name+'checkpoint.pth.tar')
    print(f"The Checkpoint for the epoch : {epoch} has been saved on the path : {dir_name}")
def set_args(__args):
    args = __args
    features_model_input = None
    if hasattr(args, 'loadModel'):
        args.model = args.loadModel.replace('.py','')

    if hasattr(args,'loadModelPruned') and args.loadModelPruned is not None and 'erfnetPruningType' in args.loadModelPruned:

        split_name = args.loadModelPruned.replace('modelPrunedCompleted/','').replace('weightsModelPrunned/','').split("/")
        args.load_dir_model_mod = split_name[0]
        features_model_input = split_name[-1].replace("(_conv_bn)","").replace("erfnetPruningType", "erfnet").replace("model_best_","").replace("non_bottleneck_1d","non bottleneck 1d").replace("(_conv)","").replace(".pth", "").split("_")

        if features_model_input is not None and not hasattr(args,'model'):
            args.model = features_model_input[0]
        if features_model_input is not None and not hasattr(args, 'loadModel'):
            args.model = features_model_input[0]
        if features_model_input is not None and (not hasattr(args,'typePruning') or args.typePruning is None):
            args.typePruning = features_model_input[1]
        if features_model_input is not None and (not hasattr(args,'pruning') or args.pruning is None):
            args.pruning = float(features_model_input[5])
        if features_model_input is not None and (not hasattr(args,'typeNorm') or args.typeNorm is None):
            args.typeNorm = int(features_model_input[3])
        if features_model_input is not None and (not hasattr(args,'listLayerPruning') or len(args.listLayerPruning)==0):
            args.listLayerPruning = [nameModule.replace(" ", "_") for nameModule in features_model_input[features_model_input.index('Layer') + 1: features_model_input.index('NumLayerPruning')]]
        if features_model_input is not None and (not hasattr(args,'listNumLayerPruning') or features_model_input[features_model_input.index('NumLayerPruning') + 1] != "ALLayer"):
            args.listNumLayerPruning = [int(numLayer) for numLayer in features_model_input[features_model_input.index('NumLayerPruning') + 1:]]
        if features_model_input is not None and (not hasattr(args,'listInnerLayerPruning') or args.listInnerLayerPruning is None):
            args.listInnerLayerPruning = ['conv', 'bn']
        if features_model_input is not None and (not hasattr(args, 'moduleErfnetPruning') or args.moduleErfnetPruning is None):
            args.moduleErfnetPruning = []

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
    if args.pruning is not None and args.pruning > 0:
        typeNorm = f"_Norm_{args.typeNorm}" if args.typeNorm else ""
        namePruning = f"PruningType_{args.typePruning}{typeNorm}_Value_{args.pruning}"
        if hasattr(args,'moduleErfnetPruning') and  len(args.moduleErfnetPruning) > 0:
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
            numberLayer = "_ALLayer"
            if len(args.listNumLayerPruning) > 0:
                numberLayer = "_NumLayerPruning"
                for number in args.listNumLayerPruning:
                    numberLayer = numberLayer + f"_{number}"

            namePruning = namePruning + numberLayer

    return args.model + ("FreezingBackbone" if hasattr(args,'freezingBackbone') else "") + (
        namePruning if hasattr(args,'pruning') and args.pruning > 0 else "")

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
    num_epochs = 10
    steps_per_epoch = len(loader_finetuning_adapting_layers)  # Numero di batch (iterazioni) per epoca
    total_steps = num_epochs * steps_per_epoch  # Numero totale di iterazioni

    # Inizializzazione dello scheduler
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    path_drive = args.path_drive + f"ModelsExtra/{args.load_dir_model_mod}/"
    checkpoint_path = path_drive + args.modelFilenameDrive.replace(".pth", "/")+'/checkpoint.pth.tar'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        print(f"Retrieving checkpoint from the path : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        if "state_dict" in checkpoint and not 'model' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))


    print("Start fine tuning pruning for adding layers ... ")
    print("Freezing layer not named : adaptingInput")
    for name, param in model.named_parameters():
        # Congela i parametri se 'adaptingInput' non è nel nome del layer
        if 'adaptingInput' not in name:
            param.requires_grad = False
        else:
            # Assicurati che i parametri che non devono essere congelati siano settati per il gradiente
            param.requires_grad = True

    for epoch in range(start_epoch, num_epochs+1):
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
            if step % 200 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        save_checkpoint_mod_on_drive({
            'epoch': epoch + 1,
            'arch': str(model),
            'scheduler': scheduler.state_dict(),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },args,epoch)

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

def show_prediction_model(output,nameFile="",saveResult=False,args=None):
    class_name = ["Road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                  "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                  "bicycle","Void"]
    color_map = {
        0: [255, 0, 0],  # Rosso
        1: [0, 255, 0],  # Verde
        2: [0, 0, 255],  # Blu
        3: [255, 255, 0],  # Giallo
        4: [255, 0, 255],  # Magenta
        5: [0, 255, 255],  # Ciano
        6: [128, 0, 0],  # Marrone
        7: [128, 128, 0],  # Oliva
        8: [0, 128, 0],  # Verde scuro
        9: [128, 0, 128],  # Viola
        10: [0, 128, 128],  # Verde acqua
        11: [0, 0, 128],  # Blu navy
        12: [255, 165, 0],  # Arancione
        13: [255, 192, 203],  # Rosa
        14: [105, 105, 105],  # Grigio
        15: [255, 69, 0],  # Rosso arancio
        16: [173, 255, 47],  # Verde-giallo
        17: [255, 215, 0],  # Oro
        18: [218, 165, 32],  # Bronzo
        19: [64, 224, 208]  # Turchese
    }

    numpy_output = output.to('cpu')
    class_output = np.argmax(numpy_output, axis=0)

    colored_image = np.zeros((class_output.shape[0], class_output.shape[1], 3), dtype=np.uint8)

    for cls in color_map:
        colored_image[class_output == cls] = color_map[cls]

    legend_handles = [mpatches.Patch(color=np.array(color) / 255, label=class_name[i]) for i, color in
                      color_map.items()]

    plt.figure(figsize=(15, 10))  # Modifica qui per cambiare le dimensioni dell'immagine
    plt.imshow(colored_image)
    plt.legend(handles=legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    if saveResult == False:
        plt.show()
    if saveResult == True:
        filename = args.modelFilenameDrive
        nameFile[1] = nameFile[1].replace('_leftImg8bit','')
        path_save = f'../save/{filename.replace(".pth", "/")}/ImagesCreated/{nameFile[0]}/{nameFile[1]}'
        if is_drive_connect():
            if not ".pth" in filename:
                filename = args.modelFilenameDrive + ".pth"
            path_drive = args.path_drive + f"ModelsExtra/{args.load_dir_model_mod}/"
            Path(path_drive).mkdir(parents=True, exist_ok=True)
            dir_name = path_drive + filename.replace(".pth", "/")+f'/ImagesCreated/{nameFile[0]}/'
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            path_save = dir_name+nameFile[1]
        else:
            Path(f'../save/{filename.replace(".pth", "/")}/ImagesCreated/{nameFile[0]}/').mkdir(parents=True, exist_ok=True)

        print(f"Saving images prediction on path : {path_save} ... ")
        plt.savefig(path_save, bbox_inches='tight')
        plt.close()
