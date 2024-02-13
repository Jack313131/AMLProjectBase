import os
import shutil
from pathlib import Path

# Percorso della directory sorgente
source_dir = Path('/Users/jacopospaccatrosi/Desktop/Polito/Advanced Machine Learning/My Final Project/Dataset Cityscapes Extra/gtCoarse/')

# Percorso della directory di destinazione
destination_dir = Path('/Users/jacopospaccatrosi/Desktop/Polito/Advanced Machine Learning/My Final Project/Dataset Cityscapes Extra/gtCoarse/gtFine')

# Crea la directory di destinazione se non esiste
destination_dir.mkdir(parents=True, exist_ok=True)

# Funzione per copiare i file con _labelTrainIds
def copy_labelTrainIds_files(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            if os.path.isdir(root+"/"+dir) and dir.startswith("gt"):
                for root2, dirs2, files2 in os.walk(root+"/"+dir):
                    for dir2 in dirs2:
                        for root3,dirs3,files3 in os.walk(root2+"/"+dir2):
                            for dir3 in dirs3:
                             for root4,dirs4,files4 in os.walk(root3+"/"+dir3):
                                for file in files4:
                                    if file.endswith('_labelTrainIds.png'):
                                        src_file_path = Path(root4) / file
                                        relative_path = src_file_path.relative_to(src_dir)
                                        dst_file_path = dst_dir / relative_path

                                        # Creazione delle sottodirectory se non esistono
                                        dst_file_path.parent.mkdir(parents=True, exist_ok=True)

                                        # Copia del file
                                        shutil.copy2(src_file_path, dst_file_path)

# Chiamata alla funzione con i percorsi sorgente e destinazione
copy_labelTrainIds_files(source_dir, destination_dir)

print(f"I file con _labelTrainIds sono stati copiati da {source_dir} a {destination_dir}.")
