
from data_functions import available_files

path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset"
all_files = available_files(root_dir=path)

###
interesting_sequences = [269, 26, 180, 288, 292, 302, 358, 363, 438, 518, 729, 750, 802, 816]

good_pairs = [
    [560, 1,3],
    [621, 4,6]
]

# Filter out only the interesting sequences
filtered_file_paths = []
for file_path in all_files:
    file_name = file_path.split("/")[-1]
    file_name_split = file_name.split("_")
    loc_id = file_name_split[2]
    if int(loc_id) in interesting_sequences:
        filtered_file_paths.append(file_path)
print("Filtered from", len(all_files), "files in sequence into", len(filtered_file_paths), " Note: CD between images outside of the sequence will not make sense!.")
