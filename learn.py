import numpy as np
from torch import nn, set_num_threads, optim, cuda, Tensor
from torchvision import transforms
from torchsummary import summary as tsummary
from os import stat
from struct import unpack
from matplotlib import pyplot as plt
from load import load_entities, load_map
from processing import init_cache, preprocess_entities, stats, prime_embedding, preprocess_map
from models import EntityEncoder


def scratch():
    entity_encoder = EntityEncoder()
    raw_entities = load_entities()
    preprocessed_entities, positional_encoding, positions = preprocess_entities(raw_entities)
    embedded_entities = prime_embedding(preprocessed_entities)
    entity_encodings, spatial_entity_encodings, encoded_entities = \
        entity_encoder.forward(embedded_entities, positional_encoding)

    raw_map = load_map()
    preprocessed_map = preprocess_map(raw_map, spatial_entity_encodings, positions, show_plots=True)
    # plt.imshow(entity_encodings.detach().numpy(), cmap='plasma')
    # plt.show()
    # plt.imshow(spatial_entity_encodings.detach().numpy(), cmap='plasma')
    # plt.show()
    

if __name__ == '__main__':
    init_cache()
    scratch()
    