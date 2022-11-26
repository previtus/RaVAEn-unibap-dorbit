
import time
import numpy as np
from data_functions import DataNormalizerLogManual, DataModule
from model_functions import Module, DeeperVAE
from util_functions import which_device, tiles2image
from anomaly_functions import twin_vae_change_score
from tqdm import tqdm
import pytorch_lightning as pl


PATH_model_weights = "/home/vitek/Vitek/Work/Trillium - RaVAEn 2/data/model/model_rgbnir.ckpt"
ROOT_data = "/home/vitek/Vitek/Work/Trillium - RaVAEn 2/data/sample_data/"
in_memory = True # True = Fast, False = Mem efficient, slow I/O

pl.seed_everything(42)

##### DATA

BANDS = [2,1,0,3,4,5,6,7,8,9] # < all 10 high res bands, ps: keeping the same order is important
# high res used:
# channels: ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
# visualisation_channels: [2, 1, 0]

# but we want:
# channels: ['B4','B3','B2', 'B8'] # RGB + NIR (in band 8) ~ all are in ground resolution of 10m
# visualisation_channels: [0, 1, 2]

BANDS = [0,1,2,6] # either
# BANDS = [2,1,0,6]# or
# does this work?

# from the original ordering > ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60','probability']

settings = {'dataloader': {
                'batch_size': 8,
                'num_workers': 4,
                'train_ratio': 1.00,
                'validation_ratio': 0.00,
                'test_ratio': 0.00,
            },
            'dataset': {
                'data_base_path': ROOT_data+'gee_after',
                'bands': BANDS, ##### < CHANGE HERE 1
                'tile_px_size': 32,
                'tile_overlap_px': 0,
                'include_last_row_colum_extra_tile': False,
                'nan_to_num': False,
             },
            'normalizer': DataNormalizerLogManual,
           }

data_normalizer = settings["normalizer"](settings)
print("loaded data_normalizer")

data_module_after = DataModule(settings, data_normalizer, in_memory)
data_module_after.setup()
data_normalizer.setup(data_module_after)
len_train_ds_after = len(data_module_after.val_dataloader())

settings_before = settings.copy()
settings_before["dataset"]["data_base_path"] = ROOT_data+'gee_before' # < this one will load from the "before" folder
data_module_before = DataModule(settings_before, data_normalizer, in_memory)
data_module_before.setup()
data_normalizer.setup(data_module_before)
len_train_ds_before = len(data_module_before.val_dataloader())

assert len_train_ds_after == len_train_ds_before


# Not sure if these are the right ones!
bands = 4
for band_i in range(bands):
    if band_i < len(data_normalizer.BANDS_S2_BRIEF):

        # rescale
        r = data_normalizer.RESCALE_PARAMS[data_normalizer.BANDS_S2_BRIEF[band_i]]
        x0,x1,y0,y1 = r["x0"], r["x1"], r["y0"], r["y1"]

        print(data_normalizer.BANDS_S2_BRIEF[band_i], "->", r)

# We can also later just use the loaded tiles:

after_array = []
before_array = []
for sample in tqdm(data_module_after.train_dataset): # IS THERE A BETTER WAY? THIS IS SLOW
    after_array.append(np.asarray(sample))
for sample in tqdm(data_module_before.train_dataset):
    before_array.append(np.asarray(sample))

before_array = np.asarray(before_array)
after_array = np.asarray(after_array)

import torch
before_array = torch.as_tensor(before_array)
after_array = torch.as_tensor(after_array)

# Note: this is something we would have in the numpy pre-processed files .npy

print("Now we have", len(before_array),"*",before_array[0].shape, "as data from the image before the event and ",len(after_array),"*",after_array[0].shape, "from the image after the event.")

# import pdb; pdb.set_trace()

#### MODEL

cfg_module = {"input_shape": (4, 32, 32), ##### < CHANGE HERE 3
              "visualisation_channels": [0, 1, 2],
              "len_train_ds": len_train_ds_after,
              "len_val_ds": 0,
}

cfg_train = {}

model_cls_args_VAE = {
        # Using Small model:
        "hidden_channels": [16, 32, 64], # number of channels after each downscale. Reversed on upscale
        "latent_dim": 128,                # bottleneck size
        "extra_depth_on_scale": 0,        # after each downscale and upscale, this many convolutions are applied
        "visualisation_channels": cfg_module["visualisation_channels"],
}

module = Module(DeeperVAE, cfg_module, cfg_train, model_cls_args_VAE)

from argparse import Namespace

hparams = {}
namespace = Namespace(**hparams)

module.load_from_checkpoint(checkpoint_path=PATH_model_weights, hparams=namespace,
                            model_cls=DeeperVAE, train_cfg=cfg_train, model_cls_args=model_cls_args_VAE)
print("loaded!")

print("list of the used model layers:")
print(module.model.encoder)
print(module.model.fc_mu)

model = module.model
model.eval()

device = which_device(model)

# We have: model.forward .encode, .decode

compare_func = twin_vae_change_score

time_total = 0
time_zero = time.time()

predicted_distances = []

# Dataloaders load it in batches - these are loaded on demand
# for before, after in zip(data_module_before.train_dataloader(), data_module_after.train_dataloader()):
# before_data = before[0]
# after_data = after[0]

# While iterating over the arrays loads only one by one - these are already loaded in memory
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
grid_shape = (32, 26)
grid_shape = (20, 20) # 400
change_map_image = tiles2image(predicted_distances, grid_shape = grid_shape, overlap=0, tile_size = 32)

import pylab as plt
plt.imshow(change_map_image[0])
plt.colorbar()
plt.show()

