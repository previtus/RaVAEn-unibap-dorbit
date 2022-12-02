import time, os
import numpy as np
from data_functions import DataNormalizerLogManual_ExtraStep, load_datamodule, file2uniqueid
from unibap_dataset_query import get_unibap_dataset_data
from model_functions import Module, DeeperVAE
from util_functions import which_device, seed_all_torch_numpy_random
from save_functions import save_latents, save_change
from anomaly_functions import encode_batch, twin_vae_change_score_from_latents
import torch
import json

# CONFIG:
BATCH_SIZE = None
NUM_WORKERS = None
in_memory = True  # True = Fast, False = Mem efficient, slow I/O
keep_latent_log_var = False # only if we want to reconstruct
# -- Keep the same: --
BANDS = [0,1,2,3] # Unibap format
LATENT_SIZE = 128
plot = False # if set to True, needs matplotlib
if plot:
    try:
        import matplotlib as plt
        from vis_functions import plot_tripple, plot_change

    except:
        print("Failed to import matplotlib, setting plot to False")
        plot = False

settings_dataloader = {'dataloader': {
                'batch_size': None,
                'num_workers': None,
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
    main_start_time = time.time()

    print("settings:", settings)

    logged = {}
    for key in settings.keys():
        logged[ "args_"+key ] = settings[key]

    BATCH_SIZE = int(settings["batch_size"])
    settings_dataloader ['dataloader']['batch_size'] = BATCH_SIZE
    NUM_WORKERS = int(settings["num_workers"])
    settings_dataloader ['dataloader']['num_workers'] = NUM_WORKERS
    SEED = int(settings["seed"])

    in_memory = not settings["special_keep_only_indices_in_mem"]
    if not in_memory: print("Careful, data is loaded with each batch, IO will be slower! (Change special_keep_only_indices_in_mem to default False if you don't want that!)")

    time_before_file_query = time.time()
    selected_files = get_unibap_dataset_data(settings)
    print("Will run on a sequence of:", selected_files)
    files_query_time = time.time() - time_before_file_query
    logged["time_files_query"] = files_query_time

    seed_all_torch_numpy_random(SEED)

    ### MODEL
    time_before_model_load = time.time()
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
    model_load_time = time.time() - time_before_model_load
    logged["time_model_load"] = model_load_time

    ### DATA


    latents_per_file = {}
    for file_i, file_path in enumerate(selected_files):
        # print("DEBUG file_i, file_path", file_i, file_path)
        previous_file = file_i - 1
        previous_file_name = selected_files[previous_file]

        previous_file_uid = file2uniqueid(previous_file_name)
        this_file_uid = file2uniqueid(file_path)
        if file_i == 0:
            previous_file_uid = "FirstFile"
        print("CD",file_i,"/",len(selected_files),":",previous_file_uid,"<>",this_file_uid)

        time_before_dataloader = time.time()
        data_module = load_datamodule(settings_dataloader, file_path, in_memory)
        tiles_n = len(data_module.train_dataset)
        dataloader = data_module.train_dataloader()
        dataloader_create = time.time() - time_before_dataloader
        logged["time_file_" + str(file_i).zfill(3) + "_dataloader_create"] = dataloader_create

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

            time_before_encode = time.time()
            mus, log_vars = encode_batch(model, batch, keep_latent_log_var)

            # print(batch.shape, "=>", mus.shape)

            batch_size = len(mus)
            latents[index:index+batch_size] = mus
            if keep_latent_log_var: latents_log_var[ index:index+batch_size ] = log_vars

            encode_time = time.time() - time_before_encode
            time_before_compare = time.time()

            if previous_file in latents_per_file:
                previous_mus = latents_per_file[previous_file][index:index+batch_size]

                distance = compare_func(mus, previous_mus)
                predicted_distances[ index:index+batch_size ] = distance
                if time_total == 0: print("Distance", distance, )

                cd_calculated = True

            time_now = time.time()
            compare_time = time_now - time_before_compare
            time_to_encode_compare = time_now - time_before_encode

            if time_total == 0:
                print("Single evaluation of batch size ",batch_size,"took ", time_to_encode_compare, " = encode batch ", encode_time, " + compare batch", compare_time)
                #print("One item from the batch ", time_to_encode_compare/batch_size, " = encode one ", encode_time/batch_size, " + compare one", compare_time/batch_size)

                logged["time_file_" + str(file_i).zfill(3) + "_first_full_batch_encode"] = encode_time
                logged["time_file_" + str(file_i).zfill(3) + "_first_full_batch_compare"] = compare_time
                logged["time_file_" + str(file_i).zfill(3) + "_first_full_batch_encode_and_compare"] = time_to_encode_compare # for sanity check mostly

            time_total += time_to_encode_compare

            index += batch_size

        time_whole_file = time.time() - time_zero

        print("Sum of all encodings and comparisons for",file_i,"took:", time_total)
        print("If we include data loading", time_whole_file)

        logged["time_file_"+ str(file_i).zfill(3) +"_total_encode_compare"] = time_total
        logged["time_file_"+ str(file_i).zfill(3) +"_total_encode_compare_with_IO"] = time_whole_file


        time_before_saves = time.time()
        save_latents(settings["results_dir"], latents, uid_name=this_file_uid)
        if keep_latent_log_var: save_latents(latents_log_var, file_i, log_var=True)

        latents_per_file[file_i] = latents

        if previous_file in latents_per_file: del latents_per_file[previous_file] # longer history no longer needed

        if cd_calculated:
            predicted_distances = np.asarray(predicted_distances).flatten()
            save_change(settings["results_dir"], predicted_distances, previous_uid_name=previous_file_uid, uid_name=this_file_uid)
            # print("DEBUG predicted_distances, previous_file, file_i", predicted_distances.shape, previous_file, file_i)
            if plot:
                # plot_change(settings["results_dir"],predicted_distances, previous_file, file_i)
                plot_tripple(settings["results_dir"],predicted_distances, previous_file, file_i, selected_files)
        time_after_saved = time.time() - time_before_saves
        logged["time_file_"+ str(file_i).zfill(3) +"_save_latents_changemap"] = time_after_saved

    # LOG
    main_end_time = time.time()
    main_run_time = (main_end_time - main_start_time)
    print("TOTAL RUN TIME = " + str(main_run_time) + "s (" + str(main_run_time / 60.0) + "min)")

    logged["time_main"] = main_run_time

    print(logged)
    with open(os.path.join(settings["results_dir"], settings["log_name"]+"_"+str(BATCH_SIZE)+"batch.json"), "w") as fh:
        json.dump(logged, fh)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Run inference')
    # parser.add_argument('--folder', default="/home/vitek/Vitek/Work/Trillium_RaVAEn_2/data/dataset of s2/unibap_dataset/",
    #                     help="Full path to local folder with Sentinel-2 files")
    parser.add_argument('--folder', default="unibap_dataset/",
                        help="Full path to local folder with Sentinel-2 files")
    parser.add_argument('--selected_images', default="all", #"all" / "tenpercent" / "first_N" / "0,1,2"
                        help="Indices to the files we want to use. Files will be processed sequentially, each pair evaluated for changes.")
    parser.add_argument('--model', default='weights/model_rgbnir',
                        help="Path to the model weights")
    parser.add_argument('--results_dir', default='results/',
                        help="Path where to save the results")
    parser.add_argument('--log_name', default='log',
                        help="Name of the log (batch size will be appended in any case).")
    # parser.add_argument('--time-limit', type=int, default=300,
    #                     help="time limit for running inference [300]")
    parser.add_argument('--batch_size', default=8,
                        help="Batch size for the dataloader and inference")
    parser.add_argument('--num_workers', default=4,
                        help="Number of workers for the dataloader, the default 4 seems to work well.")

    parser.add_argument('--seed', default=42,
                        help="Seed for torch and random calls.")

    # Keep False, unless memory explodes ...
    parser.add_argument('--special_keep_only_indices_in_mem', default=False,
                        help="Dataloader doesn't load the tiles unless asked, this will support even huge S2 scenes, but is slow.")

    args = vars(parser.parse_args())

    main(settings=args)
