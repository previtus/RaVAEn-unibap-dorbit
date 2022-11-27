import numpy as np
import math
import pylab as plt


def save_latents(latents, file_i):
    # save the dictionary of latents belonging to ith file in the sequence
    latents = latents.detach().cpu().numpy()
    path = "../results/latent_"+str(file_i).zfill(3)+".npy"

    np.save(path, latents)

def save_change(change_distances, previous_file, file_i):
    grid_size = int(math.sqrt(len(change_distances)))  # all were square
    grid_shape = (grid_size, grid_size)
    change_map_image = tiles2image(change_distances, grid_shape=grid_shape, overlap=0, tile_size=1)
    print(change_map_image.shape)

    path = "../results/pair_" + str(previous_file).zfill(3) + "-" + str(file_i).zfill(3) + "_changemap.npy"
    np.save(path, change_map_image)

def plot_change(change_distances, previous_file, file_i):
    grid_size = int(math.sqrt(len(change_distances)))  # all were square
    grid_shape = (grid_size, grid_size)
    change_map_image = tiles2image(change_distances, grid_shape=grid_shape, overlap=0, tile_size=32)

    plt.imshow(change_map_image[0])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("../results/pair_" + str(previous_file).zfill(3) + "-" + str(file_i).zfill(3) + "_result.png")
    plt.close()


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

