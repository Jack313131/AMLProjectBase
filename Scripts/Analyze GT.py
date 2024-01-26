from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Carica l'immagine di ground truth
gt_image_path = '/Users/jacopospaccatrosi/Downloads/RoadAnomaly_jpg/frames/animals04_17_animal_casualties_so_far_in_2006.labels/labels_semantic.png'
gt_image = Image.open(gt_image_path)

# Converti l'immagine in un array numpy per analizzarla
gt_array = np.array(gt_image)

# Visualizza l'immagine e l'istogramma dei valori dei pixel
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Immagine
axs[0].imshow(gt_image, cmap='gray')
axs[0].axis('off')  # Nasconde gli assi
axs[0].set_title('Ground Truth Image')

# Istogramma
axs[1].hist(gt_array.flatten(), bins=30)
axs[1].set_title('Pixel Value Distribution')

plt.show()

# Controlla anche se l'immagine è effettivamente in scala di grigi
print(f"Unique pixel values: {np.unique(gt_array)}")
