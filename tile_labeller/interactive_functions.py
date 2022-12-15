import sys
sys.path.append('../tile_classifier')
import numpy as np
from vis_functions import load_as_image, to_tile_able, show_img, tile_location, location_to_tile_id
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from os.path import exists
from six.moves import cPickle as pickle #for performance

LABELS_PATH = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/tile_annotation/"

colours = {
    -1: "gray", # unlabelled
    0: "red", # ground
    1: "blue", # cloud
}

dict_x_y_to_label = {}
dict_x_y_to_tile_i = {}
global_label_path = ""

def points_from_labels(dict_x_y_to_label):
    dot_xs = []
    dot_ys = []
    cols = []
    for x in dict_x_y_to_label.keys():
        for y in dict_x_y_to_label[x].keys():
            tile_value = dict_x_y_to_label[x][y]
            dot_xs.append(x)
            dot_ys.append(y)
            cols.append(colours[tile_value])
    return dot_xs, dot_ys, cols


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def interactive_image_with_labels(image_path, labels=None):
    global dict_x_y_to_tile_i
    global dict_x_y_to_label
    global global_label_path
    # new image, reset
    dict_x_y_to_tile_i = {}
    dict_x_y_to_label = {}

    img = load_as_image(image_path)
    img, grid_shape = to_tile_able(img)

    plot = show_img(img, show=False)
    fig = plt.gcf()

    n_tiles = int(grid_shape[0]*grid_shape[1])

    for tile_i in range(n_tiles):
        # mark with something at the tile locations ...
        dot_x, dot_y = tile_location(tile_i)

        tile_value = -1 # unlabelled
        if dot_x not in dict_x_y_to_tile_i:
            dict_x_y_to_tile_i[dot_x]={}
            dict_x_y_to_label[dot_x] = {}
        if dot_y not in dict_x_y_to_tile_i[dot_x]:
            dict_x_y_to_tile_i[dot_x][dot_y] = tile_i
            dict_x_y_to_label[dot_x][dot_y] = tile_value

    global_label_path = LABELS_PATH + image_path.split("/")[-1].split(".")[0] + ".npy"
    if exists(global_label_path):
        dict_x_y_to_label = load_dict(global_label_path)


    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print("button:", event.button)

        click_class = 1 # cloud
        click_class_str = "cloud"
        if event.button == MouseButton.RIGHT:
            click_class = 0 # ground
            click_class_str = "ground"

        elif event.button == MouseButton.MIDDLE:
            click_class = -1 # unlabelled
            click_class_str = "unlabelled"

        global dict_x_y_to_tile_i
        global dict_x_y_to_label
        tile_i_pressed, key_x, key_y = location_to_tile_id(dict_x_y_to_tile_i, ix, iy)
        print('x = %d, y = %d' % (ix, iy), "tile_i = ",tile_i_pressed, "marked as", click_class_str)
        dict_x_y_to_label[key_x][key_y] = click_class

        # global clicked_tiles
        # clicked_tiles.append([tile_i_pressed, click_class])

        # clear frame
        dot_xs, dot_ys, cols = points_from_labels(dict_x_y_to_label)
        plot.scatter(dot_xs, dot_ys, color=cols)
        plt.show()
        # plt.draw()

        return None

    def on_close(event):
        global dict_x_y_to_label
        global global_label_path
        save_dict(dict_x_y_to_label, global_label_path)
        print('Saved annotation into ', global_label_path)

    def on_keyboard(event):
        print('press', event.key)
        sys.stdout.flush()
        set_to = None

        if event.key == 'c':
            set_to = 1
        elif event.key == 'g':
            set_to = 0
        elif event.key == 'u':
            set_to = -1

        if set_to is not None:
            global dict_x_y_to_label
            for x in dict_x_y_to_label.keys():
                for y in dict_x_y_to_label[x].keys():
                    dict_x_y_to_label[x][y] = set_to

            # clear frame
            dot_xs, dot_ys, cols = points_from_labels(dict_x_y_to_label)
            plot.scatter(dot_xs, dot_ys, color=cols)
            plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('key_press_event', on_keyboard)

    dot_xs, dot_ys, cols = points_from_labels(dict_x_y_to_label)
    plot.scatter(dot_xs, dot_ys, color=cols)
    plt.show()
    # plt.draw()
