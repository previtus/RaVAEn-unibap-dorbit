import rasterio
import numpy as np

ONCE_PRINT = False
def file_to_tiles_data(filename, settings_bands = [0,1,2,3], tile_px_size=32, tile_overlap_px=0,
                       include_last_row_colum_extra_tile=False, nan_to_num=True):
    """
    Opens one tif file and extracts all tiles (given tile size and overlap).
    Returns list of the data directly.
    """
    if settings_bands is None:
        # Load all
        with rasterio.open(filename) as src:
            tile_data = src.read()
            filename_shape = src.height, src.width
    else:
        bands = [b + 1 for b in settings_bands]

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
        if nan_to_num:
            tile = np.nan_to_num(tile)
        tiles.append(tile)

    return tiles
