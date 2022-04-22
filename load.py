import numpy as np
from os import stat
from struct import unpack
from matplotlib import pyplot as plt


UNIT_ATTRIBUTE_CT = 10
UNIT_ATTRIBUTE_SZ = 2
UNIT_SZ = UNIT_ATTRIBUTE_CT * UNIT_ATTRIBUTE_SZ

TESTMAP_WIDTH = 90
TESTMAP_HEIGHT = 56
MAP_DEPTH = 4

ORDER_VALUE_CT = 4


def load_file(file_no=0):
    raw_entities = load_entities(file_no)
    if raw_entities == None:
        print(f'load::load_file(): file {file_no} error - no entities read. skipping.')
        return None

    raw_map = load_map(file_no)
    orders = load_orders(file_no)
    all_match = True

    map_frame_ct = raw_map[1].shape[0]
    order_frame_ct = len(orders)
    entity_frame_ct = len(raw_entities)

    if (map_frame_ct == order_frame_ct == entity_frame_ct):
        for i in range(map_frame_ct):
            order_shape = orders[i].shape[0]
            entity_shape = raw_entities[i].shape[0]
            if order_shape != entity_shape:
                all_match = False
                break
        if not all_match:
            print(f'load::load_file(): file {file_no} error: unequal unit counts. skipping.')
            return None
    else:
        print(f'load::load_file(): file {file_no} error: unequal frame counts. skipping.')
        return None
    
    return (raw_entities, orders, raw_map)


# TODO: subtract min x and min y from positions
def load_entities(file_no=0, test_map=False):
    if test_map:
        entity_file_path = 'data/entities.bin'
    else:
        entity_file_path = f'data/supervised/ent{file_no}.ord'

    file_size = stat(entity_file_path).st_size
    if file_size == 0:
        return None
    file_bytes = None
    with open(entity_file_path, 'rb') as entity_file:
        file_bytes = entity_file.read(file_size)
    # <: little endian, # of 2 byte items, H: unsigned short (2 byte)
    format_str = f'<{int(file_size/2)}H' 
    units = unpack(format_str, file_bytes)
    if len(units) == 0:
        return None
    
    if test_map:
        unit_ct = int(file_size / UNIT_SZ)
        return np.array([
            units[i*UNIT_ATTRIBUTE_CT:(i*UNIT_ATTRIBUTE_CT)+UNIT_ATTRIBUTE_CT]
            for i in range(unit_ct)
        ])
    else:
        cur_index = 0
        max_index = file_size // 2
        frames = []
        while True:
            try:
                unit_ct = units[cur_index]
            except:
                return None
            cur_index += 1
            frames.append(
                np.array([
                    units[cur_index + i*UNIT_ATTRIBUTE_CT: cur_index + (i*UNIT_ATTRIBUTE_CT)+UNIT_ATTRIBUTE_CT] 
                    for i in range(unit_ct)
                ], dtype=object)
            )
            cur_index += (unit_ct * UNIT_ATTRIBUTE_CT)
            if cur_index == max_index:
                break
        return frames


def load_map(file_no=0, test_map=False, visualize=False):
    if test_map:
        map_file_path = 'data/map.bin'
    else:
        map_file_path = f'data/supervised/map{file_no}.ord'

    file_size = stat(map_file_path).st_size
    if file_size == 0:
        return None
    file_bytes = None
    with open(map_file_path, 'rb') as map_file:
        file_bytes = map_file.read(file_size)
    # <: little endian, # of 1 byte items, B: unsigned byte
    format_str = f'<{int(file_size)}B' 
    unpacked_map = unpack(format_str, file_bytes)

    if test_map:
        # TODO: reformat to below w/ walkability going out separately
        width = TESTMAP_WIDTH
        height =TESTMAP_HEIGHT
        map_tiles = TESTMAP_WIDTH * TESTMAP_HEIGHT
        map_layers = [
            np.array(unpacked_map[
                i*map_tiles:(i*map_tiles)+map_tiles
            ]).reshape((height, width), order='F')
            for i in range(MAP_DEPTH)
        ]
        if visualize:
            for i in range(MAP_DEPTH):
                plt.imshow(map_layers[i], cmap='hot')
                plt.show()
        return np.array(map_layers)
    else:
        height = unpacked_map[0]
        width = unpacked_map[1]
        map_tiles = width * height

        cur_index = map_tiles + 2
        walkability = np.array(unpacked_map[2: cur_index]).reshape((height, width), order='F')

        frame_ct = (file_size - 2 - map_tiles) // map_tiles
        frames = []
        frame = np.zeros((MAP_DEPTH - 1, height, width))
        cur_frame = 0

        visibility_mask =   0b00000001
        zerg_units_mask =   0b00011100
        terran_units_mask = 0b11100000
        for i in range(frame_ct):
            for x in range(width):
                for y in range(height):
                    unpacked_map_byte = unpacked_map[cur_index]
                    frame[0][y][x] = unpacked_map_byte & visibility_mask
                    frame[1][y][x] = (unpacked_map_byte & zerg_units_mask) >> 2
                    frame[2][y][x] = (unpacked_map_byte & terran_units_mask) >> 5
                    cur_index += 1
            cur_frame += 1
            frames.append(frame)
        return walkability, np.array(frames)
        

def load_orders(file_no=0):
    order_file_path = f'data/supervised/ord{file_no}.ord'
    file_size = stat(order_file_path).st_size
    if file_size == 0:
        return None
    file_bytes = None
    with open(order_file_path, 'rb') as order_file:
        file_bytes = order_file.read(file_size)
    # <: little endian, # of 2 byte items, H: unsigned short (2 byte)
    format_str = f'<{int(file_size/2)}H' 
    orders = unpack(format_str, file_bytes)
    
    cur_index = 0
    max_index = file_size // 2
    frames = []
    while True:
        unit_ct = orders[cur_index]
        cur_index += 1
        frames.append(
            np.array([
                orders[cur_index + i*ORDER_VALUE_CT: cur_index + (i*ORDER_VALUE_CT)+ORDER_VALUE_CT] 
                for i in range(unit_ct)
            ], dtype=object)
        )
        cur_index += (unit_ct * ORDER_VALUE_CT)
        if cur_index == max_index:
            break
    return frames

