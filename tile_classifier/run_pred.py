from vis_functions import vis_image_with_predicted_labels
from dataset import available_files, file_to_tiles_data

from model_pytorch import LilModel
from dataset import tiles2latents
import numpy as np
import torch

if __name__ == "__main__":
    dataset_dir = "../unibap_dataset"

    model = LilModel()
    # tile_model_path = "../results/tile_model.pt"
    RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15_prefinal_again/"
    tile_model_path = RESULTS_DIR+"tile_model_256batch.pt"

    # # EXPERIMENT!
    # tile_model_path="/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/results/tile_model_8batch.pt"
    model.load_state_dict(torch.load(tile_model_path))
    model.eval()

    all_images = available_files(dataset_dir)
    image_ids = [path.split("/")[-1].split("_")[2] for path in all_images]

    exclude_train_set_tiles = [104, 743, 448, 127, 358, 642]

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

        Y_predictions = model(X_latents)
        print("Y predictions:", Y_predictions.shape)

        count_clouds = 0.
        count_nonclouds = 0.
        for p in Y_predictions:
            if p > 0.5: count_clouds+=1.
            else: count_nonclouds+=1.

        cloud_cover = count_clouds/(count_clouds+count_nonclouds)
        print("This image has", int(100*100 * cloud_cover)/100., "% cloud cover")

        # print(Y_predictions[0:5])

        vis_image_with_predicted_labels(image_path, Y_predictions)
        # break