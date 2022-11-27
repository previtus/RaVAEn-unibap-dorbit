import math
import time
import numpy as np
from data_functions import DataNormalizerLogManual_ExtraStep, DataModule, available_files
from model_functions import Module, DeeperVAE
from util_functions import which_device, tiles2image
from anomaly_functions import twin_vae_change_score
import pytorch_lightning as pl
from argparse import Namespace
import torch
import pylab as plt

BANDS = [0,1,2,3] # Unibap format
settings_dataloader = {'dataloader': {
                'batch_size': 8,
                'num_workers': 4,
            },
            'dataset': {
                'data_base_path': None,
                'bands': BANDS,
                'tile_px_size': 32,
                'tile_overlap_px': 0,
                'include_last_row_colum_extra_tile': False,
                'nan_to_num': False,
             },
            'normalizer': DataNormalizerLogManual_ExtraStep,
           }
cfg_module = {"input_shape": (4, 32, 32),
              "visualisation_channels": [0, 1, 2],
              "len_train_ds": 400,
              "len_val_ds": 0,
}
model_cls_args_VAE = {
        # Using Small model:
        "hidden_channels": [16, 32, 64], # number of channels after each downscale. Reversed on upscale
        "latent_dim": 128,                # bottleneck size
        "extra_depth_on_scale": 0,        # after each downscale and upscale, this many convolutions are applied
        "visualisation_channels": cfg_module["visualisation_channels"],
}

############################################################################


def main(settings):
    print("settings:", settings)

    ### DATA
    pl.seed_everything(42)
    in_memory = True  # True = Fast, False = Mem efficient, slow I/O

    settings_dataloader["dataset"]["data_base_path"] = settings['after']

    data_normalizer = settings_dataloader["normalizer"](settings_dataloader)
    data_module_after = DataModule(settings_dataloader, data_normalizer, in_memory)
    data_module_after.setup()
    data_normalizer.setup(data_module_after)

    settings_dataloader_before = settings_dataloader.copy()
    settings_dataloader_before["dataset"]["data_base_path"] = settings['before']
    data_module_before = DataModule(settings_dataloader_before, data_normalizer, in_memory)
    data_module_before.setup()
    data_normalizer.setup(data_module_before)

    ### MODEL
    cfg_train = {}
    module = Module(DeeperVAE, cfg_module, cfg_train, model_cls_args_VAE)
    hparams = {}
    namespace = Namespace(**hparams)

    module.load_from_checkpoint(checkpoint_path=settings["model"], hparams=namespace,
                                model_cls=DeeperVAE, train_cfg=cfg_train, model_cls_args=model_cls_args_VAE)
    print("Loaded model!")

    ### LATENTS
    model = module.model
    model.eval()

    device = which_device(model)
    compare_func = twin_vae_change_score
    time_total = 0
    time_zero = time.time()

    predicted_distances = []

    after_array = []
    before_array = []
    for sample in data_module_after.train_dataset:
        after_array.append(np.asarray(sample))
    for sample in data_module_before.train_dataset:
        before_array.append(np.asarray(sample))
    before_array = np.asarray(before_array)
    after_array = np.asarray(after_array)
    before_array = torch.as_tensor(before_array).float()
    after_array = torch.as_tensor(after_array).float()

    for before, after in zip(before_array, after_array):
        before_data = before.unsqueeze(0)
        after_data = after.unsqueeze(0)

        start_time = time.time()

        distances = compare_func(model, before_data, after_data)
        predicted_distances.append(distances)

        if time_total == 0: print("Distances", distances, )

        end_time = time.time()
        single_eval = (end_time - start_time)
        if time_total == 0: print("Single evaluation took ", single_eval)
        time_total += single_eval

    time_end = time.time() - time_zero

    print("Full evaluation took", time_total)
    print("If we include data loading", time_end)

    predicted_distances = np.asarray(predicted_distances).flatten()

    grid_size = int(math.sqrt(len(predicted_distances))) # all were square
    grid_shape = (grid_size, grid_size)
    change_map_image = tiles2image(predicted_distances, grid_shape = grid_shape, overlap=0, tile_size = 32)

    plt.imshow(change_map_image[0])
    plt.colorbar()
    plt.savefig("../results/result.png")


if __name__ == "__main__":
    import argparse
    s2files = available_files("../unibap_dataset/")
    assert len(s2files) > 0, f"Not found files in S2 folder!!"

    parser = argparse.ArgumentParser('Run inference')
    parser.add_argument('--before', default=s2files[0],
                        help="Full path to local Sentinel-2 file corresponding to the earlier tile")
    parser.add_argument('--after', default=s2files[1],
                        help="Full path to local Sentinel-2 file corresponding to the later tile")
    parser.add_argument('--model', default='../weights/model_rgbnir.ckpt',
                        help="name of the model weights")
    # parser.add_argument('--time-limit', type=int, default=300,
    #                     help="time limit for running inference [300]")

    args = vars(parser.parse_args())

    main(settings=args)

