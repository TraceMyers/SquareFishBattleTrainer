import numpy as np
from torch import nn, set_num_threads, optim, cuda, Tensor
from torchvision import transforms
from torchsummary import summary as tsummary
from os import stat
from struct import unpack
from matplotlib import pyplot as plt
from load import load_entities
from processing import init_cache, preprocessed_entities, stats
from models import EntityEncoder


def scratch():
    # TODO: make 
    entities, positional_encoding = preprocessed_entities(load_entities())
    entity_encoder = EntityEncoder(stats('entity_col_ct'))
    entity_encoder.add_to_batch(entities, positional_encoding)
    entity_encoder.add_to_batch(entities, positional_encoding)
    entity_encoder.add_to_batch(entities, positional_encoding)
    entity_encoder.forward()


if __name__ == '__main__':
    init_cache()
    scratch()
    