import numpy as np
import random, torch
def which_device(model):
    device = next(model.parameters()).device
    print("Model is on:", device)
    return device

def seed_all_torch_numpy_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sequence2pairs(sequence):
    pairs = []
    for i in range(len(sequence)-1):
        pairs.append([sequence[i], sequence[i+1]])
    return pairs
def tiles2image(predicted_distances, grid_shape, overlap=0, tile_size = 32, channels = 1):
    # predicted_distances shape of ~ N
    image = np.zeros((channels, grid_shape[1]*tile_size, grid_shape[0]*tile_size), dtype=np.float32)
    index = 0
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            tile = predicted_distances[index] * np.ones((channels, tile_size, tile_size))
            image[:, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
            index += 1
    return image

