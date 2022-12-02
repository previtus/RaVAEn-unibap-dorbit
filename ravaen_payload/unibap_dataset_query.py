from data_functions import available_files

def get_interesting_pairs(all_files, good_pairs):
    # good_pairs = [
    #     [560, 1, 3],
    #     [621, 4, 6]
    # ]
    selected_location_ids = [pair[0] for pair in good_pairs]

    paths_by_location_id = {}
    for file_path in all_files:
        file_name = file_path.split("/")[-1]
        file_name_split = file_name.split("_")
        loc_id = int(file_name_split[2])
        if loc_id in selected_location_ids:
            if loc_id not in paths_by_location_id:
                paths_by_location_id[loc_id] = []
            paths_by_location_id[loc_id].append(file_path)

    filtered_file_paths = []
    last_one = ""
    for pair in good_pairs:
        loc_id, first_i, second_i = pair

        if str(loc_id)+"_"+str(first_i) == last_one:
            pass # we can skip this one as it's already in the sequence
        else:
            first = paths_by_location_id[loc_id][first_i]
            filtered_file_paths.append(first)

        second = paths_by_location_id[loc_id][second_i]
        filtered_file_paths.append(second)

        last_one = str(loc_id)+"_"+str(second_i)

    return filtered_file_paths


def get_interesting_sequences_only(all_files, interesting_sequences):
    # Filter out only the interesting sequences
    filtered_file_paths = []
    for file_path in all_files:
        file_name = file_path.split("/")[-1]
        file_name_split = file_name.split("_")
        loc_id = file_name_split[2]
        if int(loc_id) in interesting_sequences:
            filtered_file_paths.append(file_path)
    return filtered_file_paths

def get_unibap_dataset_data(settings):
    files_sequence = available_files(settings["folder"])
    if settings["unibap_dataset_filter"] == "none":
        print("Kept original ordering of the data and all", len(files_sequence), "items in the sequence.")
    if settings["unibap_dataset_filter"] == "sequences100":
        interesting_sequences = [269, 26, 180, 288, 292, 302, 358, 363, 438, 518, 729, 750, 802, 816]
        files_sequence = get_interesting_sequences_only(files_sequence, interesting_sequences)
        print("Filtered down to", len(files_sequence), "items in the sequence.")
    elif settings["unibap_dataset_filter"] == "pairs60":
        from good_pairs import good_pairs_60
        files_sequence = get_interesting_pairs(files_sequence, good_pairs_60)
        print("Filtered down to", len(files_sequence), "items in the sequence.")
    elif settings["unibap_dataset_filter"] == "pairs37":
        from good_pairs import good_pairs_37
        files_sequence = get_interesting_pairs(files_sequence, good_pairs_37)
        print("Filtered down to", len(files_sequence), "items in the sequence.")

    if settings["selected_images"] == "all":
        selected_idx = [i for i in range(len(files_sequence))] # all selected
    elif settings["selected_images"] == "tenpercent":
        # select just 10 percent of the images - from 1024 into just 102
        # still good enough sample... but faster
        ten_percent = int(len(files_sequence) / 10.)
        selected_idx = [i for i in range(ten_percent)]  # sample
    elif settings["selected_images"].startswith("first_"):
        first_n = settings["selected_images"].split("first_")[-1]
        try:
            first_n = int(first_n)
        except:
            print("failed with parsing how many first images, reverting to just 10")
            first_n = 10
        selected_idx = [i for i in range(first_n)]
    else:
        selected_idx = [int(idx) for idx in settings["selected_images"].split(",")]
    assert len(selected_idx) <= len(files_sequence), f"Selected more indices than how many we have images!"
    selected_files = []
    for idx in selected_idx:
        selected_files.append(files_sequence[idx])

    return selected_files