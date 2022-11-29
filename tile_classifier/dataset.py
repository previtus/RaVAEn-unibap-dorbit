import glob, os
from vis_functions import *

def available_files(root_dir="/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset"):
    return sorted(glob.glob(os.path.join(root_dir,"*.tif")))

def ids2path(loc_id, seq_num):
    all_files = available_files()
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
    loc_id = 527
    seq_num = 5
    ids =    [0,1,2,3,4]
    labels = [1,1,1,0,0]

    ids = "all"
    labels = "allclouds" # 1 cloud, 0 no-cloud

    if ids == "all":
        ids = [i for i in range(n_tiles)]
        labels = [1 for i in range(n_tiles)]
    # all or a list ...

    item = [loc_id, seq_num, ids, labels]

    items = [item]



    return items

def visualize_tiles_with_annotations(tile_indices):
    for item in tile_indices:
        loc_id, seq_num, ids, labels = item
        image_path = ids2path(loc_id, seq_num)

        #vis_image(image_path)
        vis_image_with_tile_labels(image_path, ids, labels)

        break

# Get tile indices
# each image has 255 tiles, one index should be [ "img path", ids ]
tile_indices = demo_tile_indices()

visualize_tiles_with_annotations(tile_indices)

# # Get a dataset
# X,Y = get_dataset("unibap_dataset", tile_indices)