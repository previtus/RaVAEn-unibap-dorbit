from data_functions import available_files

def get_unibap_dataset_data(settings):
    files_sequence = available_files(settings["folder"])

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