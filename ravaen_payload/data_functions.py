import glob
import rasterio
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ONCE_PRINT = True

def available_files(root_dir="."):
    return sorted(glob.glob(os.path.join(root_dir,"*.tif")))

def available_result_files(root_dir=".", ending="npy"):
    return sorted(glob.glob(os.path.join(root_dir,"*."+ending)))


def find_file_path_from_uid(all_files, id="110", time="04-19-2019"):
    for idx, file_name in enumerate(all_files):
        unique_name_id = file2uniqueid(file_name)
        l = unique_name_id.split("_")
        id_, time_ = l[1], l[3]
        if id==id_ and time == time_:
            # found it
            return idx
    print("Didn't find the corresponding file!")
    return None

def file2uniqueid(file_name = "eopatch_id_109_col_7_row_16_07-30-2019_etcetc.tif"):
    if "/" in file_name:
        file_name = file_name.split("/")[-1]

    if not file_name.startswith("eopatch_id_"):
        print("Warning, the file path doesnt start with the expected string eopatch_id_, things may break!")
    file_name_split = file_name.split("_")
    loc_id = file_name_split[2]
    time_id = file_name_split[7]
    unique_name_id = "id_" + loc_id + "_time_" + time_id
    return unique_name_id

def load_all_tile_indices_from_folder(settings_dataset):
    path = settings_dataset["data_base_path"]

    isDirectory = os.path.isdir(path)
    if isDirectory:
        # A directory, load all tifs inside
        allfiles = glob.glob(path + "/*.tif")
        allfiles.sort()
    elif ".tif" in path:
        # A single file, load that one directly
        allfiles = [path]

    tiles = []

    for idx, filename in enumerate(allfiles):
        tiles_from_file = file_to_tiles_indices(filename, settings_dataset,
                                                tile_px_size=settings_dataset["tile_px_size"],
                                                tile_overlap_px=settings_dataset["tile_overlap_px"],
                                                include_last_row_colum_extra_tile=settings_dataset[
                                                    "include_last_row_colum_extra_tile"])

        tiles += tiles_from_file
        print(idx, filename, "loaded", len(tiles_from_file), "tiles.")

    print("Loaded:", len(tiles), "total tile indices")
    return tiles

def load_all_tile_indices_from_folder(settings_dataset):
    path = settings_dataset["data_base_path"]

    isDirectory = os.path.isdir(path)
    if isDirectory:
        # A directory, load all tifs inside
        allfiles = glob.glob(path + "/*.tif")
        allfiles.sort()
    elif ".tif" in path:
        # A single file, load that one directly
        allfiles = [path]

    tiles = []

    for idx, filename in enumerate(allfiles):
        tiles_from_file = file_to_tiles_indices(filename, settings_dataset,
                                                tile_px_size=settings_dataset["tile_px_size"],
                                                tile_overlap_px=settings_dataset["tile_overlap_px"],
                                                include_last_row_colum_extra_tile=settings_dataset[
                                                    "include_last_row_colum_extra_tile"])

        tiles += tiles_from_file
        print(idx, filename, "loaded", len(tiles_from_file), "tiles.")

    print("Loaded:", len(tiles), "total tile indices")
    return tiles

def file_to_tiles_indices(filename, settings, tile_px_size=128, tile_overlap_px=4,
                          include_last_row_colum_extra_tile=True):
    """
    Opens one tif file and extracts all tiles (given tile size and overlap).
    Returns list of indices to the tile (to postpone in memory loading).
    """

    with rasterio.open(filename) as src:
        filename_shape = src.height, src.width

    data_h, data_w = filename_shape
    if data_h < tile_px_size or data_w < tile_px_size:
        # print("skipping, too small!")
        return []

    h_tiles_n = int(np.floor((data_h - tile_overlap_px) / (tile_px_size - tile_overlap_px)))
    w_tiles_n = int(np.floor((data_w - tile_overlap_px) / (tile_px_size - tile_overlap_px)))

    tiles = []
    for h_idx in range(h_tiles_n):
        for w_idx in range(w_tiles_n):
            tiles.append([w_idx * (tile_px_size - tile_overlap_px), h_idx * (tile_px_size - tile_overlap_px)])
    if include_last_row_colum_extra_tile:
        for w_idx in range(w_tiles_n):
            tiles.append([w_idx * (tile_px_size - tile_overlap_px), data_h - tile_px_size])
        for h_idx in range(h_tiles_n):
            tiles.append([data_w - tile_px_size, h_idx * (tile_px_size - tile_overlap_px)])
        tiles.append([data_w - tile_px_size, data_h - tile_px_size])

    # Save file ID + corresponding tiles[]
    tiles_indices = [[filename] + t + [tile_px_size, tile_px_size] for t in tiles]
    return tiles_indices

def load_all_tile_data_from_folder(settings_dataset):
    path = settings_dataset["data_base_path"]

    isDirectory = os.path.isdir(path)
    if isDirectory:
        # A directory, load all tifs inside
        allfiles = glob.glob(path + "/*.tif")
        allfiles.sort()
    elif ".tif" in path:
        # A single file, load that one directly
        allfiles = [path]

    tiles = []

    for idx, filename in enumerate(allfiles):
        tiles_from_file = file_to_tiles_data(filename, settings_dataset,
                                                tile_px_size=settings_dataset["tile_px_size"],
                                                tile_overlap_px=settings_dataset["tile_overlap_px"],
                                                include_last_row_colum_extra_tile=settings_dataset[
                                                    "include_last_row_colum_extra_tile"])

        tiles += tiles_from_file
        print(idx, filename, "loaded", len(tiles_from_file), "tiles.")

    print("Loaded:", len(tiles), "total tile indices")
    return tiles

def file_to_tiles_data(filename, settings, tile_px_size=128, tile_overlap_px=4,
                          include_last_row_colum_extra_tile=True):
    """
    Opens one tif file and extracts all tiles (given tile size and overlap).
    Returns list of the data directly.
    """
    if settings['bands'] is None:
        # Load all
        with rasterio.open(filename) as src:
            tile_data = src.read()
            filename_shape = src.height, src.width
    else:
        bands = [b + 1 for b in settings['bands']]

        global ONCE_PRINT
        if ONCE_PRINT:
            print("DEBUG - loaded bands", bands)
            ONCE_PRINT = False
        with rasterio.open(filename) as src:
            tile_data = src.read(bands)
            filename_shape = src.height, src.width

    data_h, data_w = filename_shape
    if data_h < tile_px_size or data_w < tile_px_size:
        # print("skipping, too small!")
        return []

    h_tiles_n = int(np.floor((data_h - tile_overlap_px) / (tile_px_size - tile_overlap_px)))
    w_tiles_n = int(np.floor((data_w - tile_overlap_px) / (tile_px_size - tile_overlap_px)))

    tiles_indices = []
    for h_idx in range(h_tiles_n):
        for w_idx in range(w_tiles_n):
            tiles_indices.append([w_idx * (tile_px_size - tile_overlap_px), h_idx * (tile_px_size - tile_overlap_px)])
    if include_last_row_colum_extra_tile:
        for w_idx in range(w_tiles_n):
            tiles_indices.append([w_idx * (tile_px_size - tile_overlap_px), data_h - tile_px_size])
        for h_idx in range(h_tiles_n):
            tiles_indices.append([data_w - tile_px_size, h_idx * (tile_px_size - tile_overlap_px)])
        tiles_indices.append([data_w - tile_px_size, data_h - tile_px_size])

    tiles = []
    for tile_idx in tiles_indices:
        y, x = tile_idx
        w, h = tile_px_size, tile_px_size
        tile = tile_data[:, x:x+w, y:y+h]
        tile = np.float32(tile)
        if settings['nan_to_num']:
            tile = np.nan_to_num(tile)
        tiles.append(tile)

    return tiles

def load_tile_idx(tile, settings):
    """
    Loads tile data values from the saved indices (file and window locations).
    """
    filename, x, y, w, h = tile

    # load window:
    window = rasterio.windows.Window(row_off=y, col_off=x, width=w, height=h)

    if settings['bands'] is None:
        # Load all
        with rasterio.open(filename) as src:
            tile_data = src.read(window=window)
    else:
        bands = [b + 1 for b in settings['bands']]

        global ONCE_PRINT
        if ONCE_PRINT:
            print("DEBUG - loaded bands", bands)
            ONCE_PRINT = False
        with rasterio.open(filename) as src:
            tile_data = src.read(bands, window=window)

    tile_data = np.float32(tile_data)
    if settings['nan_to_num']:
        tile_data = np.nan_to_num(tile_data)

    return tile_data

class DataNormalizerLogManual():

    def __init__(self, settings):
        self.normalization_parameters = None

    def setup(self, data_module):
        # These were edited to work with the 10 bands we have
        # only use 10m resolution bands (10): Blue (B2), Green (B3), Red (B4), VNIR (B5),
        # VNIR (B6), VNIR (B7), NIR (B8), VNIR (B8a), SWIR (B11), SWIR (B12) combining
        # all 10
        # self.BANDS_S2_BRIEF = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        # RGB+NIR
        self.BANDS_S2_BRIEF = ["B4", "B3", "B2", "B8"]  ##### < CHANGE HERE 2

        self.RESCALE_PARAMS = {
            "B1": {"x0": 7.3,
                   "x1": 7.6,
                   "y0": -1,
                   "y1": 1,
                   },
            "B2": {"x0": 6.9,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B3": {"x0": 6.5,
                   "x1": 7.4,
                   "y0": -1,
                   "y1": 1,
                   },
            "B4": {"x0": 6.2,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B5": {"x0": 6.1,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B6": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B7": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B8": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B8A": {"x0": 6.5,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    },
            "B9": {"x0": 6,
                   "x1": 7,
                   "y0": -1,
                   "y1": 1,
                   },
            "B10": {"x0": 2.5,
                    "x1": 4.5,
                    "y0": -1,
                    "y1": 1,
                    },
            "B11": {"x0": 6,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    },
            "B12": {"x0": 6,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    }
        }
        print("normalization params are manually found")

    def normalize_x(self, data):
        bands = data.shape[0]  # for example 15
        for band_i in range(bands):
            data_one_band = data[band_i, :, :]
            if band_i < len(self.BANDS_S2_BRIEF):
                # log
                data_one_band = np.log(data_one_band)
                data_one_band[np.isinf(data_one_band)] = np.nan

                # rescale
                r = self.RESCALE_PARAMS[self.BANDS_S2_BRIEF[band_i]]
                x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
                data_one_band = ((data_one_band - x0) / (x1 - x0)) * (y1 - y0) + y0
            data[band_i, :, :] = data_one_band
        return data

    def denormalize_x(self, data):
        bands = data.shape[0]  # for example 15
        for band_i in range(bands):
            data_one_band = data[band_i, :, :]
            if band_i < len(self.BANDS_S2_BRIEF):
                # rescale
                r = self.RESCALE_PARAMS[self.BANDS_S2_BRIEF[band_i]]
                x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
                data_one_band = (((data_one_band - y0) / (y1 - y0)) * (x1 - x0)) + x0

                # undo log
                data_one_band = np.exp(data_one_band)
                # data_one_band = np.log(data_one_band)
                # data_one_band[np.isinf(data_one_band)] = np.nan

            data[band_i, :, :] = data_one_band
        return data


def match_stats_abcd(B, a, b, c, d):
    return (B - a) / (b - a) * (d - c) + c


def match_GEE(B):
    band0 = match_stats_abcd(B[0], 96, 1401, 235.0, 3422.0)
    band1 = match_stats_abcd(B[1], 172, 1315, 427.0, 3213.0)
    band2 = match_stats_abcd(B[2], 264, 1347, 645.0, 3290.0)
    band3 = match_stats_abcd(B[3], 158, 2165, 401.0, 5288.0)
    """
    band 0 a,b,c,d 96 , 1401 , 235.0 , 3422.0
    band 1 a,b,c,d 172 , 1315 , 427.0 , 3213.0
    band 2 a,b,c,d 264 , 1347 , 645.0 , 3290.0
    band 3 a,b,c,d 158 , 2165 , 401.0 , 5288.0
    """

    img_m = [band0, band1, band2, band3]
    img_m = np.asarray(img_m)
    # img_m = torch.stack(img_m)
    return img_m

class DataNormalizerLogManual_ExtraStep(DataNormalizerLogManual):
    def __init__(self, settings):
        super().__init__(settings)

    def normalize_x(self, data):
        data = match_GEE(data.astype(float))
        bands = data.shape[0]  # for example 15

        for band_i in range(bands):
            data_one_band = data[band_i, :, :]
            if band_i < len(self.BANDS_S2_BRIEF):
                # log
                data_one_band = np.log(data_one_band)
                data_one_band[np.isinf(data_one_band)] = np.nan

                # rescale
                r = self.RESCALE_PARAMS[self.BANDS_S2_BRIEF[band_i]]
                x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
                data_one_band = ((data_one_band - x0) / (x1 - x0)) * (y1 - y0) + y0
            data[band_i, :, :] = data_one_band
        return data

class TileDataset(Dataset):
    # Main class that holds a dataset with smaller tiles originally extracted from larger geotiff files
    # (Optionally) Minimal impact on memory, loads actual data of x only in __getitem__ (when loading a batch of data)
    def __init__(self, tiles, settings_dataset, data_normalizer=None, in_memory = False):
        self.tiles = tiles
        self.settings_dataset = settings_dataset
        self.data_normalizer = data_normalizer

        self.in_memory = in_memory

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x = self.tiles[idx]
        # Load only when needed:

        if not self.in_memory:
            # If not loaded fully in memory, we have to load it here
            x = load_tile_idx(x, self.settings_dataset)

        if self.data_normalizer is not None:
            x = self.data_normalizer.normalize_x(x)

        x = torch.from_numpy(x).float()
        return x

class DataModule(torch.nn.Module): # torch.nn.Module # pl.LightningDataModule
    # if we set in_memory to True, it loads the data faster, but the memory may run out. If set to False, the actual
    # data loading occurs only when tasked.
    def __init__(self, settings, data_normalizer, in_memory=False):
        super().__init__()
        self.settings = settings
        self.data_normalizer = data_normalizer

        self.batch_size = self.settings["dataloader"]["batch_size"]
        self.num_workers = self.settings["dataloader"]["num_workers"]

        self.train_ratio = 1.0
        self.validation_ratio = 0.0
        self.test_ratio = 0.0

        self.in_memory = in_memory

        self.setup_finished = False

    def prepare_data(self):
        # Could contain data download and unpacking...
        pass

    def setup(self, stage=None):
        if self.setup_finished:
            return True  # to prevent double setup

        if not self.in_memory:
            tiles = load_all_tile_indices_from_folder(self.settings["dataset"])
            print("Altogether we have", len(tiles), "tiles (loaded as indices).")
        else:
            tiles = load_all_tile_data_from_folder(self.settings["dataset"])
            print("Altogether we have", len(tiles), "tiles (loaded directly).")

        if self.train_ratio == 1.0:
            tiles_train = tiles
            tiles_test = []
            tiles_val = []
        else:
            tiles_train, tiles_rest = train_test_split(tiles, test_size=1 - self.train_ratio)
            tiles_val, tiles_test = train_test_split(tiles_rest, test_size=self.test_ratio / (
                        self.test_ratio + self.validation_ratio))

        print("train, test, val:", len(tiles_train), len(tiles_test), len(tiles_val))

        self.train_dataset = TileDataset(tiles_train, self.settings["dataset"], self.data_normalizer, self.in_memory)
        self.test_dataset = TileDataset(tiles_test, self.settings["dataset"], self.data_normalizer, self.in_memory)
        self.val_dataset = TileDataset(tiles_val, self.settings["dataset"], self.data_normalizer, self.in_memory)

        self.setup_finished = True

    def train_dataloader(self):
        """Initializes and returns the training dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self, num_workers=None):
        """Initializes and returns the validation dataloader"""
        num_workers = num_workers or self.num_workers
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=num_workers)

    def test_dataloader(self, num_workers=None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=num_workers)



class DataModuleDummy(DataModule): # torch.nn.Module # pl.LightningDataModule
    # if we set in_memory to True, it loads the data faster, but the memory may run out. If set to False, the actual
    # data loading occurs only when tasked.
    def __init__(self, settings, data_normalizer, in_memory=False):
        super().__init__(settings, data_normalizer, in_memory)

    def prepare_data(self):
        # Could contain data download and unpacking...
        pass

    def setup(self, num_tiles, number_of_bands, stage=None):
        if self.setup_finished:
            return True  # to prevent double setup

        tiles = [np.random.rand(number_of_bands, 32, 32) for i in range(num_tiles)]
        print("Altogether we have", len(tiles), "tiles (randomized dummies).")

        tiles_train = tiles
        tiles_test = []
        tiles_val = []

        print("train, test, val:", len(tiles_train), len(tiles_test), len(tiles_val))

        self.train_dataset = TileDataset(tiles_train, self.settings["dataset"], self.data_normalizer, self.in_memory)
        self.test_dataset = TileDataset(tiles_test, self.settings["dataset"], self.data_normalizer, self.in_memory)
        self.val_dataset = TileDataset(tiles_val, self.settings["dataset"], self.data_normalizer, self.in_memory)

        self.setup_finished = True


def load_datamodule(settings_dataloader, file_path, in_memory):
    # load data of this file
    settings_dataloader_local = settings_dataloader.copy()
    settings_dataloader_local["dataset"]["data_base_path"] = file_path

    data_normalizer = settings_dataloader["normalizer"](settings_dataloader)
    data_module = DataModule(settings_dataloader_local, data_normalizer, in_memory)
    data_module.setup()
    data_normalizer.setup(data_module)

    return data_module # data_module.train_dataloader()

def create_dummy_data_module_v2(settings_dataloader, file_path, in_memory, num_tiles = 225, number_of_bands = 4):
    # load data of this file
    settings_dataloader_local = settings_dataloader.copy()
    settings_dataloader_local["dataset"]["data_base_path"] = file_path

    data_normalizer = settings_dataloader["normalizer"](settings_dataloader)
    data_module = DataModuleDummy(settings_dataloader_local, data_normalizer, in_memory)
    data_module.setup(num_tiles, number_of_bands)
    data_normalizer.setup(data_module)

    return data_module # data_module.train_dataloader()


def create_dummy_dataloader_v1(settings_dataloader, num_tiles = 225, number_of_bands = 4):
    settings_dataloader_local = settings_dataloader.copy()
    print(settings_dataloader_local)

    tile_size = settings_dataloader_local['dataset']['tile_px_size']
    data = np.random.rand( num_tiles, number_of_bands, tile_size, tile_size )
    import torch
    data = torch.from_numpy(data).float() # default float gets interpretted as tensor Double
    print("Made up random data:", data.shape)

    num_workers = settings_dataloader_local['dataloader']['num_workers']
    batch_size = settings_dataloader_local['dataloader']['batch_size']
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers), num_tiles

def load_data_array_with_dataloaders(settings_dataloader, file_path, in_memory):
    # load data of this file
    settings_dataloader_local = settings_dataloader.copy()
    settings_dataloader_local["dataset"]["data_base_path"] = file_path

    data_normalizer = settings_dataloader["normalizer"](settings_dataloader)
    data_module_local = DataModule(settings_dataloader_local, data_normalizer, in_memory)
    data_module_local.setup()
    data_normalizer.setup(data_module_local)

    data_array = []
    for sample in data_module_local.train_dataset:
        data_array.append(np.asarray(sample))
    data_array = np.asarray(data_array)
    data_array = torch.as_tensor(data_array).float()

    return data_array

def load_data_array_simple(settings_dataloader, file_path):
    tiles = file_to_tiles_data(file_path, settings_dataloader["dataset"], tile_px_size=32, tile_overlap_px=0, include_last_row_colum_extra_tile=False)
    data_normalizer = DataNormalizerLogManual_ExtraStep(None)
    data_normalizer.setup(None)

    tiles = np.asarray(tiles)

    # tiles2image_DEBUG(tiles)

    data_array = [data_normalizer.normalize_x(tile) for tile in tiles]
    data_array = np.asarray(data_array)

    # tiles2image_DEBUG(data_array, denormalise=True)

    data_array = torch.as_tensor(data_array).float()

    return data_array


def tiles2image_DEBUG(tiles, denormalise = False):
    print("debug showing as", tiles.shape, tiles.dtype)
    print("rages min,mean,max", np.min(tiles), np.mean(tiles), np.max(tiles))

    if denormalise:
        data_normalizer = DataNormalizerLogManual_ExtraStep(None)
        data_normalizer.setup(None)
        tiles = [data_normalizer.denormalize_x(tile) for tile in tiles]
        tiles = np.asarray(tiles)

    tiles_n, channels, tile_size, _ = tiles.shape
    side = int(math.sqrt(tiles_n))  # 15

    grid_shape = (side, side)
    image = np.zeros((channels, grid_shape[1] * tile_size, grid_shape[0] * tile_size), dtype=np.float32)
    index = 0
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            tile = tiles[index]
            image[:, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = tile
            index += 1
    print("image", image.shape)  # reconstruction has lost some sizes ...

    image_for_plot = np.moveaxis(image[0:3], 0, -1)
    image_for_plot = image_for_plot / 2000.0
    print("shape, min, max", image_for_plot.shape, np.min(image_for_plot), np.max(image_for_plot))
    # image_for_plot = image_for_plot * 1000.0

    import pylab as plt
    plt.imshow(image_for_plot)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.close()
