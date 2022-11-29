import sys
sys.path.append('../ravaen_payload')

import math
import time
import numpy as np
from data_functions import DataNormalizerLogManual_ExtraStep, DataNormalizerLogManual, load_data_array_with_dataloaders, load_data_array_simple, available_files
from model_functions import Module, DeeperVAE
from util_functions import which_device, seed_all_torch_numpy_random
from save_functions import save_latents, save_change, plot_change
from anomaly_functions import encode_tile, twin_vae_change_score_from_latents
from argparse import Namespace
import torch

BANDS = [0,1,2,3] # Unibap format
LATENT_SIZE = 128
keep_latent_log_var = False
# if we want to reconstruct the results, then we need them... but for just distances we don't care
plot = False # if set to True, needs matplotlib

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

def prep():
    model_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/weights/" + "model_rgbnir"

    seed_all_torch_numpy_random(42)
    ### MODEL
    cfg_train = {}
    module = Module(DeeperVAE, cfg_module, cfg_train, model_cls_args_VAE)
    module.model.encoder.load_state_dict(torch.load(model_path + "_encoder.pt"))
    module.model.fc_mu.load_state_dict(torch.load(model_path + "_fc_mu.pt"))
    if keep_latent_log_var:
        module.model.fc_var.load_state_dict(torch.load(model_path + "_fc_var.pt"))

    print("Loaded model!")
    module.model.eval()
    model = module.model
    # device = which_device(model)
    return model

model_global = None

def get_model():
    global model_global
    if model_global is None:
        model_global = prep()
        return model_global
    else:
        return model_global

def encode(tile_data):
    latent = None

    model = get_model()

    tile_data = torch.from_numpy(tile_data).float()

    latent_mu, latent_log_var = encode_tile(model, tile_data, keep_latent_log_var=False)
    latent = latent_mu.detach().cpu().numpy()

    return latent