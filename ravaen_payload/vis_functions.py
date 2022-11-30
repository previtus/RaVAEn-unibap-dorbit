import rasterio
import math
import numpy as np
from save_functions import tiles2image
import os

def plot_change(save_dir, change_distances, previous_file, file_i):
    grid_size = int(math.sqrt(len(change_distances)))  # all were square
    grid_shape = (grid_size, grid_size)
    change_map_image = tiles2image(change_distances, grid_shape=grid_shape, overlap=0, tile_size=32)

    import pylab as plt
    plt.imshow(change_map_image[0])
    plt.colorbar()
    plt.tight_layout()

    path = os.path.join(save_dir, "pair_" + str(previous_file).zfill(3) + "-" + str(file_i).zfill(3) + "_result.png")
    plt.savefig(path)
    plt.close()

def plot_tripple(save_dir, change_distances, previous_file, file_i, file_paths):
    # before / after
    before = load_as_image(file_paths[previous_file])
    after = load_as_image(file_paths[file_i])
    before, _ = to_tile_able(before)
    after, _ = to_tile_able(after)
    # change map
    grid_size = int(math.sqrt(len(change_distances)))  # all were square
    grid_shape = (grid_size, grid_size)
    change_map_image = tiles2image(change_distances, grid_shape=grid_shape, overlap=0, tile_size=32)

    print("before:", before.shape)
    print("after:", before.shape)
    print("change_map_image:", before.shape)

    plot = show_imgs([before, after, change_map_image], show=False)
    plot.tight_layout()
    path = os.path.join(save_dir, "pair_" + str(previous_file).zfill(3) + "-" + str(file_i).zfill(3) + "_result_tripple.png")
    plot.savefig(path)
    plot.close()

def load_as_image(image_path):
    print(image_path)

    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3])  # rgb
        print(img.shape)
    img = img.astype(float)
    return img

def to_tile_able(img, tile_size=32):
    _, w, h = img.shape
    w_times = int(math.floor(w / tile_size))
    h_times = int(math.floor(h / tile_size))
    img = img[:, 0:int(w_times*tile_size), 0:int(h_times*tile_size) ]
    print("tileable as", img.shape)
    return img, [w_times,h_times]


def show_imgs(imgs, show=True):
    import rasterio.plot as rstplt
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(imgs), figsize=(3 * 5, 5), tight_layout=True)
    for ax_i, img in enumerate(imgs):
        img = np.clip(img / 2000., 0, 1)  # 3000
        rstplt.show(img, ax=ax[ax_i])
        ax[ax_i].axis("off")
    if show: plt.show()
    return plt

