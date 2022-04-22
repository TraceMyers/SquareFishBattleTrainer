import numpy as np
from torch import nn, set_num_threads, optim, cuda, Tensor
from torchvision import transforms
from torchsummary import summary as tsummary
from os import stat
from struct import unpack
from matplotlib import pyplot as plt
from load import load_entities, load_map, load_orders, load_file
from processing import init_cache, preprocess_entities, stats, prime_embedding, preprocess_map, transformer_positional_encoding
from models import EntityEncoder, SpatialEncoder
from pickle import load as pload, dump as pdump


FILE_RANGE_MAX = 358


def pickle_file(file_no=0):
    file_data = load_file(file_no)
    if file_data is not None:
        with open('data/pickled_data.bin', 'wb') as f:
            pdump(file_data, f)
        print(f'learn::pickle_file(): file [{file_no}] pickled.')
    else:
        print(f'learn::pickle_file(): bad data file [{file_no}]. not pickling.')


def pickled_file_testing(frame_no=1000):
    file_data = None
    try:
        with open('data/pickled_data.bin', 'rb') as f:
            file_data = pload(f, encoding='bytes')
    except:
        print(f'learn::pickled_file_testing(): could not read pickled file. returning.')
        return

    entity_frames = file_data[0]
    order_frames = file_data[1]
    walkability = file_data[2][0][None, :, :]
    map_frames = file_data[2][1]

    raw_entities = entity_frames[frame_no]
    raw_orders = order_frames[frame_no]
    raw_map = np.append(map_frames[frame_no], walkability, axis=0)
    time_encoding = transformer_positional_encoding(frame_no, 128)

    entity_encoder = EntityEncoder()
    preprocessed_entities, positional_encoding, positions = preprocess_entities(raw_entities)
    embedded_entities = prime_embedding(preprocessed_entities)
    entity_encodings, spatial_entity_encodings, encoded_entities = \
        entity_encoder.forward(embedded_entities, positional_encoding)

    spatial_encoder = SpatialEncoder()
    preprocessed_map = preprocess_map(
        raw_map, 
        spatial_entity_encodings.detach().numpy(), 
        positions
    )
    encoded_spatial, map_skips = spatial_encoder.forward(preprocessed_map)


def train():
    pass
    

if __name__ == '__main__':
    init_cache()
    # pickle_file(0)
    pickled_file_testing()
    
    