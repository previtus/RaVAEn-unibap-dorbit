import os, glob
from data_functions import available_files, find_file_path_from_uid
from vis_functions import plot_change, plot_tripple
import numpy as np
def load_changemap(changemap_path):
    change_map_image = np.load(changemap_path)
    return change_map_image

RESULTS_DIR = "../results"
# RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results05_newnames_allversions/"
# RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results08_selected26files/"
RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results10reducedfin/"

RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results00_dec6/"

# RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results13d_modelVersionsAndPlottable/" # from a openvino cpu and mu only model

# is only onnx enough?
# RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results13g_isJustOnnxFileEnough/"

RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15_prefinal_again/"

path = "../unibap_dataset"
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
