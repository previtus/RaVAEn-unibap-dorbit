import time, os
import numpy as np
from data_functions import DataNormalizerLogManual_ExtraStep, load_datamodule, available_files
from model_functions import Module, DeeperVAE
from util_functions import which_device, seed_all_torch_numpy_random
from save_functions import save_latents, save_change
from vis_functions import plot_tripple, plot_change
from anomaly_functions import encode_batch, twin_vae_change_score_from_latents
import torch
import json

# CONFIG:
BATCH_SIZE = 8
NUM_WORKERS = 4
in_memory = True  # True = Fast, False = Mem efficient, slow I/O
keep_latent_log_var = False # only if we want to reconstruct
# -- Keep the same: --
BANDS = [0,1,2,3] # Unibap format
LATENT_SIZE = 128
plot = True # if set to True, needs matplotlib
if plot: import matplotlib as plt

settings_dataloader = {'dataloader': {
                'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS,
            },
            'dataset': {
                'bands': BANDS, 'data_base_path': None,
                'tile_px_size': 32, 'tile_overlap_px': 0, 'include_last_row_colum_extra_tile': False, 'nan_to_num': False,
             },
            'normalizer': DataNormalizerLogManual_ExtraStep,
           }
cfg_module = {"input_shape": (4, 32, 32), "visualisation_channels": [0, 1, 2], "len_train_ds": 1, "len_val_ds": 1,}
model_cls_args_VAE = {
        # Using Small model:
        "hidden_channels": [16, 32, 64], # number of channels after each downscale. Reversed on upscale
        "latent_dim": LATENT_SIZE,                # bottleneck size
        "extra_depth_on_scale": 0,        # after each downscale and upscale, this many convolutions are applied
        "visualisation_channels": cfg_module["visualisation_channels"],
}

############################################################################


def main(settings):
    print("settings:", settings)


    logged = {}
    for key in settings.keys():
        logged[ "args_"+key ] = settings[key]

    start_time = time.time()
    files_sequence = available_files(settings["folder"])

    if settings["selected_images"] == "all":
        selected_idx = [i for i in range(len(files_sequence))] # all selected
    else:
        selected_idx = [int(idx) for idx in settings["selected_images"].split(",")]
    assert len(selected_idx) <= len(files_sequence), f"Selected more indices than how many we have images!"
    selected_files = []
    for idx in selected_idx:
        selected_files.append(files_sequence[idx])

    print("Will run on a sequence of:", selected_files)
    files_query_time = time.time() - start_time
    logged["time_files_query"] = files_query_time

    seed_all_torch_numpy_random(42)

    ### MODEL
    start_time = time.time()
    cfg_train = {}
    module = Module(DeeperVAE, cfg_module, cfg_train, model_cls_args_VAE)
    module.model.encoder.load_state_dict(torch.load(settings["model"]+"_encoder.pt"))
    module.model.fc_mu.load_state_dict(torch.load(settings["model"]+"_fc_mu.pt"))
    if keep_latent_log_var:
        module.model.fc_var.load_state_dict(torch.load(settings["model"]+"_fc_var.pt"))

    print("Loaded model!")
    module.model.eval()
    model = module.model
    # device = which_device(model)
    model_load_time = time.time() - start_time
    logged["time_model_load"] = model_load_time

    ### DATA


    latents_per_file = {}
    for file_i, file_path in enumerate(selected_files):
        # print("DEBUG file_i, file_path", file_i, file_path)
        previous_file = file_i - 1

        start_time = time.time()
        data_module = load_datamodule(settings_dataloader, file_path, in_memory)
        tiles_n = len(data_module.train_dataset)
        dataloader = data_module.train_dataloader()
        if file_i == 0:
            dataloader_create = time.time() - start_time
            logged["time_dataloader_create_first_file"] = dataloader_create

        # get latents and save them
        # use them to calculate change map in comparison with the previous image in sequence

        compare_func = twin_vae_change_score_from_latents
        time_total = 0
        time_zero = time.time()

        predicted_distances = np.zeros(tiles_n)
        cd_calculated = False
        latents = torch.zeros((tiles_n,LATENT_SIZE))
        if keep_latent_log_var: latents_log_var = torch.zeros((tiles_n,LATENT_SIZE))

        index = 0
        for batch in dataloader:

            start_time = time.time()
            mus, log_vars = encode_batch(model, batch, keep_latent_log_var)

            # print(batch.shape, "=>", mus.shape)

            batch_size = len(mus)
            latents[index:index+batch_size] = mus
            if keep_latent_log_var: latents_log_var[ index:index+batch_size ] = log_vars
            encode_time = time.time() - start_time

            if previous_file in latents_per_file:
                previous_mus = latents_per_file[previous_file][index:index+batch_size]

                distance = compare_func(mus, previous_mus)
                predicted_distances[ index:index+batch_size ] = distance
                if time_total == 0: print("Distance", distance, )

                cd_calculated = True

            compare_time = (time.time() - start_time) - encode_time
            end_time = time.time()
            single_eval = (end_time - start_time)
            if time_total == 0:
                print("Single evaluation took ", single_eval, " = encode batch ", encode_time, " + compare batch", compare_time)
                print("One item from the batch ", single_eval/batch_size, " = encode one ", encode_time/batch_size, " + compare one", compare_time/batch_size)

                logged["time_file_" + str(file_i).zfill(3) + "_first_full_batch_encode"] = encode_time
                logged["time_file_" + str(file_i).zfill(3) + "_first_full_batch_compare"] = compare_time

            time_total += single_eval

            index += batch_size

        save_latents(settings["results_dir"], latents, file_i)
        if keep_latent_log_var: save_latents(latents_log_var, file_i, log_var=True)

        latents_per_file[file_i] = latents

        if previous_file in latents_per_file: del latents_per_file[previous_file] # longer history no longer needed

        time_end = time.time() - time_zero

        print("Full evaluation of file",file_i,"took", time_total)
        print("If we include data loading", time_end)

        logged["time_file_"+ str(file_i).zfill(3) +"_total_encode_compare"] = time_total
        logged["time_file_"+ str(file_i).zfill(3) +"_total_encode_compare_with_IO"] = time_end

        if cd_calculated:
            predicted_distances = np.asarray(predicted_distances).flatten()
            save_change(settings["results_dir"], predicted_distances, previous_file, file_i)
            # print("DEBUG predicted_distances, previous_file, file_i", predicted_distances.shape, previous_file, file_i)
            if plot:
                # plot_change(settings["results_dir"],predicted_distances, previous_file, file_i)
                # todo: plot with the two tiles as well for better viz?
                plot_tripple(settings["results_dir"],predicted_distances, previous_file, file_i, selected_files)

    # LOG
    print(logged)
    with open(os.path.join(settings["results_dir"], "logs.json"), "w") as fh:
        json.dump(logged, fh)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Run inference')
    # parser.add_argument('--folder', default="/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset/",
    #                     help="Full path to local folder with Sentinel-2 files")
    parser.add_argument('--folder', default="../unibap_dataset_small/",
                        help="Full path to local folder with Sentinel-2 files")
    parser.add_argument('--selected_images', default="all", #"all" / "0,1,2"
                        help="Indices to the files we want to use. Files will be processed sequentially, each pair evaluated for changes.")
    parser.add_argument('--model', default='../weights/model_rgbnir',
                        help="Path to the model weights")
    parser.add_argument('--results_dir', default='../results/',
                        help="Path to the model weights")
    # parser.add_argument('--time-limit', type=int, default=300,
    #                     help="time limit for running inference [300]")

    args = vars(parser.parse_args())

    main(settings=args)

