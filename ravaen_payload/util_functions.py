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
