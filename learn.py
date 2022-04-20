import numpy as np
from torch import nn, set_num_threads, optim, cuda, Tensor
from torchvision import transforms
from torchsummary import summary as tsummary
from os import stat
from struct import unpack
from matplotlib import pyplot as plt
from load import load_entities, load_map
from processing import init_cache, preprocess_entities, stats, prime_embedding, preprocess_map
from models import EntityEncoder, SpatialEncoder


def scratch():
    entity_encoder = EntityEncoder()
    raw_entities = load_entities()
    preprocessed_entities, positional_encoding, positions = preprocess_entities(raw_entities)
    embedded_entities = prime_embedding(preprocessed_entities)
    entity_encodings, spatial_entity_encodings, encoded_entities = \
        entity_encoder.forward(embedded_entities, positional_encoding)

    spatial_encoder = SpatialEncoder()
    raw_map = load_map()
    preprocessed_map = preprocess_map(
        raw_map, 
        spatial_entity_encodings.detach().numpy(), 
        positions
    )
    encoded_spatial, map_skips = spatial_encoder.forward(preprocessed_map)
    for i in range(4):
        plt.imshow(map_skips[i].detach().numpy()[0], cmap='plasma')
        plt.show()
    

if __name__ == '__main__':
    init_cache()
    scratch()
    