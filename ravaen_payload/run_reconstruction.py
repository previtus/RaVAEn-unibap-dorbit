import math
import time
import numpy as np
from data_functions import DataNormalizerLogManual_ExtraStep, load_data_array_with_dataloaders, available_result_files
from model_functions import Module, DeeperVAE
from util_functions import which_device, seed_all_torch_numpy_random
from save_functions import save_latents, save_change, plot_change
from anomaly_functions import encode_tile, twin_vae_change_score_from_latents
from argparse import Namespace
import torch
import pylab as plt

BANDS = [0,1,2,3] # Unibap format
LATENT_SIZE = 128
keep_latent_log_var = True # if we want to reconstruct the results, then we need them... then keep to True

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
              "len_train_ds": 1, "len_val_ds": 1,
}
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

    result_files = available_result_files(settings["folder"])

    print("Will run on a sequence of:", result_files)

    seed_all_torch_numpy_random(42)

    ### MODEL
    cfg_train = {}
    module = Module(DeeperVAE, cfg_module, cfg_train, model_cls_args_VAE)
    module.model.load_state_dict(torch.load(settings["model"]), strict=False)

    print("Loaded model!")
    module.model.eval()
    model = module.model
    # device = which_device(model)

    latent_mus = [f for f in result_files if "logvar" not in f]
    latent_logvar = [f for f in result_files if "logvar" in f]

    assert len(latent_mus) == len(latent_logvar), f"Need the same number of latents for mus and logvars!"

    print(latent_mus)
    print(latent_logvar)

    for latent_i in range(len(latent_mus)):
        mus, logvars = np.load(latent_mus[latent_i]), np.load(latent_logvar[latent_i])

        reconstructions = []
        for tile_i in range(len(mus)):
            mu, log_var = mus[tile_i], logvars[tile_i]

            mu = torch.as_tensor(mu).float()
            log_var = torch.as_tensor(log_var).float()

            z = model.reparameterize(mu, log_var)
            reconstruction = model.decode(z)
            reconstruction = reconstruction.detach().cpu().numpy()
            reconstructions.append(reconstruction)

        reconstructions = np.asarray(reconstructions)
        print("reconstruction for latent", latent_i, "we get", reconstructions.shape)
        # (225, 1, 4, 32, 32) > into a preview image ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Run inference')
    parser.add_argument('--folder', default="../results/",
                        help="Path to results folder")
    parser.add_argument('--model', default='../_model resaved/model_rgbnir.ckpt',
                        help="Full model weights")

    args = vars(parser.parse_args())

    main(settings=args)

