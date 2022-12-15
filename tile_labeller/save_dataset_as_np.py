from read_labels import dataset_from_manual_annotation
import numpy as np

### DATASET:
PATH = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/tile_annotation/l1/"
X_latents, _, Y = dataset_from_manual_annotation(PATH)
print("Dataset:")
print("X latents:", X_latents.shape)
print("Y labels:", Y.shape)

np.savez_compressed('dataset_as_np_v2manuallyclicked_', X_latents=X_latents, Y=Y)
