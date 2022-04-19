import numpy as np
from os import stat
from struct import unpack
from matplotlib import pyplot as plt


UNIT_ATTRIBUTE_CT = 10
UNIT_ATTRIBUTE_SZ = 2
UNIT_SZ = UNIT_ATTRIBUTE_CT * UNIT_ATTRIBUTE_SZ

MAP_WIDTH = 90
MAP_HEIGHT = 56
MAP_TILES = MAP_WIDTH * MAP_HEIGHT
MAP_DEPTH = 4
MAP_BYTES = MAP_TILES * MAP_DEPTH


def load_entities():
    entity_file_path = 'data/entities.bin'
    file_size = stat(entity_file_path).st_size
    file_bytes = None
    with open(entity_file_path, 'rb') as entity_file:
        file_bytes = entity_file.read(file_size)
    unit_ct = int(file_size / UNIT_SZ)
    # <: little endian, # of 2 byte items, H: unsigned short (2 byte)
    format_str = f'<{int(file_size/2)}H' 
    units = unpack(format_str, file_bytes)
    return np.array([
        units[i*UNIT_ATTRIBUTE_CT:(i*UNIT_ATTRIBUTE_CT)+UNIT_ATTRIBUTE_CT] 
        for i in range(unit_ct)
    ])


def load_map(visualize=False):
    map_file_path = 'data/map.bin'
    file_size = stat(map_file_path).st_size
    file_bytes = None
    with open(map_file_path, 'rb') as map_file:
        file_bytes = map_file.read(file_size)
    # <: little endian, # of 1 byte items, B: unsigned byte
    format_str = f'<{int(file_size)}B' 
    unpacked_map = unpack(format_str, file_bytes)
    map_layers = [
        np.array(unpacked_map[
            i*MAP_TILES:(i*MAP_TILES)+MAP_TILES
        ]).reshape((MAP_HEIGHT, MAP_WIDTH), order='F')
        for i in range(MAP_DEPTH)
    ]
    if visualize:
        for i in range(MAP_DEPTH):
            plt.imshow(map_layers[i], cmap='hot')
            plt.show()
    return np.array(map_layers)

