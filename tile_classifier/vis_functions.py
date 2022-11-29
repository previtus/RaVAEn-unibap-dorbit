import rasterio
import rasterio.plot as rstplt
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm
import os
import math

colours = {
    0: "blue",
    1: "red",
}

from debug_comparable import *

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

def show_img(img, show=True):
    fig, ax = plt.subplots(1, 1, figsize=(3 * 5, 5), tight_layout=True)
    img = np.clip(img / 3000., 0, 1)  # 3000
    rstplt.show(img, ax=ax)
    ax.axis("off")
    if show: plt.show()

    return plt

def show_imgs(imgs, show=True):
    fig, ax = plt.subplots(1, len(imgs), figsize=(3 * 5, 5), tight_layout=True)
    for ax_i, img in enumerate(imgs):
        img = np.clip(img / 3000., 0, 1)  # 3000
        rstplt.show(img, ax=ax[ax_i])
        ax[ax_i].axis("off")
    if show: plt.show()
    return plt


def vis_image(image_path):
    img = load_as_image(image_path)
    img, grid_shape = to_tile_able(img)
    show_img(img)

def tile_location(tile_id, tile_size=32, grid_shape=[15,15]):
    offset_into_center = int(tile_size / 2)

    h_tiles_n = grid_shape[0]
    w_tiles_n = grid_shape[1]
    index = 0
    for h_idx in range(h_tiles_n):
        for w_idx in range(w_tiles_n):
            if index == tile_id:
                return w_idx * tile_size + offset_into_center, h_idx * tile_size + offset_into_center
            index += 1

def vis_image_with_tile_labels(image_path, ids, labels):
    # check!
    dataloader_tiles = file_to_tiles_data(image_path)

    img = load_as_image(image_path)
    img, grid_shape = to_tile_able(img)
    plt = show_img(img, show=False)
    print("got ids, labels=", ids, labels)

    dot_xs = []
    dot_ys = []
    cols = []
    debug_tiles = []
    for i, tile_id in enumerate(ids):
        dataloader_tile = dataloader_tiles[tile_id][0:3].astype(float)
        print(tile_id, "is", dataloader_tile.shape)
        debug_tiles.append(dataloader_tile)

        # mark with something at the tile locations ...
        dot_x, dot_y = tile_location(tile_id)
        dot_xs.append(dot_x)
        dot_ys.append(dot_y)
        cols.append(colours[labels[i]])

    plt.scatter(dot_xs, dot_ys, color=cols)
    # plt.show()
    plt.draw()


    show_imgs(debug_tiles)