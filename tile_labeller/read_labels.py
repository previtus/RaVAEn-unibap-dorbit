import glob, os
from dataset import available_files, file_to_tiles_data, tiles2latents
import numpy as np
import torch
from interactive_functions import load_dict
from vis_functions import tile_location

def available_annotations(root_dir="."):
    return sorted(glob.glob(os.path.join(root_dir,"*.npy")))

def dataset_from_manual_annotation(annotation_folder, exclude_train_set_tiles=[]):
    # dataset
    dataset_dir = "../unibap_dataset"
    all_images = available_files(dataset_dir)
    image_ids = [path.split("/")[-1].split("_")[2] for path in all_images]
    all_images_files = [path.split("/")[-1].replace(".tif", "") for path in all_images]

    # annotations
    all_labels = available_annotations(annotation_folder)
    all_labels_files = [path.split("/")[-1].replace(".npy", "") for path in all_labels]

    print("Found", len(all_images), "images and ", len(all_labels), "annotation files.")
    # print(all_images[0:5])
    # print(all_labels[0:5])

    dict_image_to_labels = {}
    for img_path in all_images_files:
        if img_path in all_labels_files:
            idx = all_labels_files.index(img_path)
            dict_image_to_labels[ img_path ] = all_labels[idx]

    # print("Images with labels:",dict_image_to_labels)

    # we want:
    #X latents: (1305, 128)
    #Y labels: (1305,)
    #X latents: torch.Size([1305, 128]) torch.float32
    #Y labels: torch.Size([1305, 1]) torch.float32

    accumulated_X = []
    accumulated_Y = []

    for i, image_path in enumerate(all_images):
        image_id = image_ids[i]
        image_key = all_images_files[i]

        if image_key not in dict_image_to_labels:
            # not labelled
            continue

        if int(image_id) in exclude_train_set_tiles:
            # on purpose ignored ...
            print(image_id, "ignored")

            continue

        # Run the model on this image's tiles ...
        X_tiles = file_to_tiles_data(image_path)
        X_tiles = np.asarray(X_tiles).astype(float)

        # now load labels:
        labels_path = dict_image_to_labels[image_key]
        dict_x_y_to_label = load_dict(labels_path)

        ## uhuh
        annotations = {}
        n_tiles = len(X_tiles)
        for tile_i in range(n_tiles):
            x, y = tile_location(tile_i)
            v = dict_x_y_to_label[x][y]
            if v != -1: # ignore the unlabelleded tiles
                annotations[tile_i] = v

        for tile_i in annotations.keys():
            accumulated_X.append(X_tiles[tile_i])
            accumulated_Y.append(annotations[tile_i])

    accumulated_X = np.asarray(accumulated_X)
    accumulated_Y = np.asarray(accumulated_Y)
    print("data tiles, labels:", accumulated_X.shape, accumulated_Y.shape)

    accumulated_latents = tiles2latents(accumulated_X)

    return accumulated_latents, accumulated_X, accumulated_Y

if __name__ == "__main__":

    PATH = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/tile_annotation/l1/"
    X_latents, X_tiles, Y = dataset_from_manual_annotation(PATH)

    print("Dataset:")
    print("X latents:", X_latents.shape)
    print("X tiles:", X_tiles.shape)
    print("Y labels:", Y.shape)



    exclude_train_set_tiles = [104, 743, 448, 127, 358, 642]
    X_latents_test, X_tiles_test, Y_test = dataset_from_manual_annotation(PATH, exclude_train_set_tiles)

    print("Dataset:")
    print("X latents (test):", X_latents_test.shape)
    print("X tiles (test):", X_tiles_test.shape)
    print("Y labels (test):", Y_test.shape)


