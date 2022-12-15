import sys
sys.path.append('../tile_classifier')

from interactive_functions import interactive_image_with_labels
from dataset import available_files, file_to_tiles_data
from dataset import tiles2latents
import numpy as np
import torch

if __name__ == "__main__":
    dataset_dir = "../unibap_dataset"

    all_images = available_files(dataset_dir)
    image_ids = [path.split("/")[-1].split("_")[2] for path in all_images]

    # exclude_train_set_tiles = [104, 743, 448, 127, 358, 642]
    exclude_train_set_tiles = []

    for i, image_path in enumerate(all_images):
        image_id = image_ids[i]
        if image_id in exclude_train_set_tiles:
            continue

        # Run the model on this image's tiles ...
        X_tiles = file_to_tiles_data(image_path)
        X_tiles = np.asarray(X_tiles).astype(float)

        X_latents = tiles2latents(X_tiles)
        X_latents = torch.from_numpy(X_latents).float()
        print("X latents:", X_latents.shape)

        labels = None
        interactive_image_with_labels(image_path, labels)
        # break