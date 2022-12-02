import os, glob
from data_functions import available_files, find_file_path_from_uid
from vis_functions import plot_change, plot_tripple
import numpy as np
def load_changemap(changemap_path):
    change_map_image = np.load(changemap_path)
    return change_map_image

RESULTS_DIR = "../results"
path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset"
all_files = available_files(root_dir=path)

change_map_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_changemap.npy")))

for i, change_map_file in enumerate(change_map_files):
    filename = change_map_file.split("/")[-1]
    filename_list = filename.split("_")

    pair_before_id = filename_list[2]
    pair_before_time = filename_list[4]
    before_file_i = find_file_path_from_uid(all_files, pair_before_id, pair_before_time)

    pair_after_id = filename_list[6]
    pair_after_time = filename_list[8]
    after_file_i = find_file_path_from_uid(all_files, pair_after_id, pair_after_time)

    print(before_file_i,"<>", after_file_i)
    ch = load_changemap(change_map_file)
    predicted_distances = np.asarray(ch).flatten()
    print(ch.shape, "=>", predicted_distances.shape)

    #plot_change("../plots", predicted_distances, i, i+1)
    plot_tripple("../plots", predicted_distances, before_file_i, after_file_i, all_files)
