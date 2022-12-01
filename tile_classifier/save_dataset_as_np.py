from dataset import generate_dataset
import numpy as np

### DATASET:
X_latents, _, Y = generate_dataset()
print("Dataset:")
print("X latents:", X_latents.shape)
print("Y labels:", Y.shape)

np.savez_compressed('dataset_as_np', X_latents=X_latents, Y=Y)
