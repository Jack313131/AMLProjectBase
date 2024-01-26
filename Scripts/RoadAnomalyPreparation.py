import os
from pathlib import Path
import shutil

# Definisci il percorso di partenza
start_path = '/Users/jacopospaccatrosi/Downloads/RoadAnomaly/frames/'

# Lista per mantenere i percorsi dei file .webp
webp_files = []

# Lista per mantenere i percorsi delle cartelle
directories = []

Path(start_path+f"finalDaset/").mkdir(parents=True, exist_ok=True)
Path(start_path+f"finalDaset/GT").mkdir(parents=True, exist_ok=True)
Path(start_path+f"finalDaset/images").mkdir(parents=True, exist_ok=True)

# Cammina attraverso la struttura delle directory
for root, dirs, files in os.walk(start_path):

    webp_files.extend([f for f in files if f.endswith('.webp')])

    for file in webp_files:
        print(f"Processing the file {file}")
        pathGT = root+ file.replace("webp", "labels")
        if os.path.isdir(pathGT):
            fileGT = file.replace("webp","png")
            shutil.copy2(pathGT+"/labels_semantic.png", start_path+f"/finalDaset/GT/{fileGT}")
            shutil.copy2(root+file, start_path+f"finalDaset/images/{file}")
        else:
            print(f"The path {pathGT} for the file {file} has not been found")
