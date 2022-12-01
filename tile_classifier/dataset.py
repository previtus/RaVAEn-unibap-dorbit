import glob, os
import numpy as np
from standalone_encoder import encode
from debug_comparable import file_to_tiles_data

def available_files(root_dir="."):
    return sorted(glob.glob(os.path.join(root_dir,"*.tif")))

def ids2path(loc_id, seq_num, dataset_dir="unibap_dataset"):
    all_files = available_files(dataset_dir)
    location_files = []

    # f_ids = []
    for path in all_files:
        f = path.split("/")[-1].split("_")
        # debug print all
        #print(f[2], "==", f[4]+"_"+f[6])

        # f_ids.append(f[2])
        if str(loc_id) == f[2]:
            location_files.append(path)

    # print(location_files[seq_num])
    return location_files[seq_num]

def demo_tile_indices(n_tiles = 225):
    def tiles_without_k_lines(k, line_n = 15):
        return [i for i in range(k*line_n,n_tiles)]

    items = []
    # cloudy
    items.append( [104, 3, "all", "allclouds"] )
    items.append( [743, 6, tiles_without_k_lines(2), [1 for i in range(len(tiles_without_k_lines(2)))] ] )
    items.append( [448, 2, tiles_without_k_lines(1), [1 for i in range(len(tiles_without_k_lines(1      )))] ] )
    # non-cloudy
    items.append( [127, 2, "all", "allnonclouds"] )
    items.append( [358, 1, "all", "allnonclouds"] )
    items.append( [642, 1, "all", "allnonclouds"] )

    for item in items:
        if item[2] == "all": item[2] = [i for i in range(n_tiles)]
        if item[3] == "allclouds": item[3] = [1 for i in range(n_tiles)]
        if item[3] == "allnonclouds": item[3] = [0 for i in range(n_tiles)]

    return items

def visualize_tiles_with_annotations(tile_indices, dataset_dir):
    from vis_functions import vis_image_with_tile_labels

    for item in tile_indices:
        loc_id, seq_num, ids, labels = item
        image_path = ids2path(loc_id, seq_num, dataset_dir)

        #vis_image(image_path)
        vis_image_with_tile_labels(image_path, ids, labels)


def select_tiles_from_image(image_path, ids):
    dataloader_tiles = file_to_tiles_data(image_path)

    tiles = []
    for i, tile_id in enumerate(ids):
        dataloader_tile = dataloader_tiles[tile_id].astype(float)
        # print(tile_id, "is", dataloader_tile.shape)
        tiles.append(dataloader_tile)

    return tiles
def get_dataset_tiles(tile_indices, dataset_dir):
    all_tiles = []
    all_labels = []
    for item in tile_indices:
        loc_id, seq_num, ids, labels = item
        image_path = ids2path(loc_id, seq_num, dataset_dir)
        tiles = select_tiles_from_image(image_path, ids)
        all_tiles += tiles
        all_labels += labels
    all_tiles = np.asarray(all_tiles)
    all_labels = np.asarray(all_labels)
    print("data tiles, labels:", all_tiles.shape, all_labels.shape)
    return all_tiles, all_labels

def tiles2latents(tiles):
    latents = []
    for tile in tiles:
        l = encode(tile)
        latents.append(l)
    latents = np.asarray(latents)
    print("encoded as", latents.shape)
    return latents

def generate_dataset(vis_dataset=False):
    # Get tile indices
    # each image has 255 tiles, one index should be [ "img path", ids ]
    tile_indices = demo_tile_indices()

    dataset_dir = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset"

    if vis_dataset: visualize_tiles_with_annotations(tile_indices, dataset_dir)

    # # Get a dataset
    X_tiles, Y = get_dataset_tiles(tile_indices, dataset_dir)

    X_latents = tiles2latents(X_tiles)

    return X_latents, X_tiles, Y

if __name__ == "__main__":

    X_latents, X_tiles, Y = generate_dataset()
    print("Dataset:")
    print("X latents:", X_latents.shape)
    print("X tiles:", X_tiles.shape)
    print("Y labels:", Y.shape)
