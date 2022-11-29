from vis_functions import *
from standalone_encoder import *
from dataset import available_files

from model_pytorch import LilModel
from dataset import tiles2latents

if __name__ == "__main__":
    dataset_dir = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset"

    model = LilModel()
    tile_model_path = "tile_model.pt"
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

        # print(Y_predictions[0:5])

        vis_image_with_predicted_labels(image_path, Y_predictions)
        # break