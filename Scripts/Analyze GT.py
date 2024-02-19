import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Percorso della directory contenente le immagini di ground truth
gt_images_directory = '/Users/jacopospaccatrosi/Desktop/Polito/Advanced Machine Learning/My Final Project/AMLProjectBase/dataset2/Validation_Dataset/FS_LostFound_full/labels_masks'

# Elenco di tutti i file nella directory
gt_image_files = os.listdir(gt_images_directory)

# Cicla su ogni file di immagine
for gt_image_file in gt_image_files:
    gt_image_path = os.path.join(gt_images_directory, gt_image_file)
    gt_image = Image.open(gt_image_path)
    gt_array = np.array(gt_image)

    try:
        # Visualizza l'immagine e l'istogramma dei valori dei pixel
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(gt_image, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title(f'Ground Truth Image - {gt_image_file}')
        axs[1].hist(gt_array.flatten(), bins=30)
        axs[1].set_title('Pixel Value Distribution')
        plt.show()
    except:
        continue

    # Stampa i valori unici dei pixel
    print(f"Unique pixel values in {gt_image_file}: {np.unique(gt_array)}")
