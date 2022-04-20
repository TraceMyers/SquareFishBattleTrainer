"""
The models in this file were made possible due to the work done by DeepMind on their project 
Alphastar. The models are largely translated and modified from those models described by DeepMind.
"""

from torch import nn, cuda, Tensor, matmul, mean as tmean, sum as tsum, flatten as tflatten
from torchvision import transforms
from torchsummary import summary as tsummary
import numpy as np


"""
Module with built-in skip connection
"""
class ResModule(nn.Module):

    def __init__(self, module):
        super(ResModule, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


"""
The encoder-half of a transformer with three blocks of two-headed self-attention. 

Inputs:
Entities (Starcraft units visible this frame) embedded into vectors of length head_dim*head_ct and 
positional encodings of each unit with the same dimension.
Outputs:
Long entity encodings, shorter entity encodings that become inputs to the spatial encoder, and a 
single vector encoding that represents all units this frame.

Where Alphastar's Entity Encoder uses one-hots for attributes plus a concatenated 
binary-encoded position as input, Sendai's does a prime number-based embedding of the 
concatenated attribute one-hots to conform the size of the attribute vector to the size of the 
encoder. This simplifies skip-connections after attention heads. It also uses the kind of positional 
encoding typical of Transformers, but adapted for 2d (x, y) input. The implementations of those 
encoders are in processing.py.
"""
class EntityEncoder(nn.Module):

    def __init__(self, head_ct=2, head_dim=128, depth=3, spatial_entity_encoding_sz=8):
        super(EntityEncoder, self).__init__()

        self.atten_dim = head_dim*head_ct
        assert(self.atten_dim % spatial_entity_encoding_sz == 0)
        seesz = spatial_entity_encoding_sz

        self.head_ct = head_ct
        self.head_dim = head_dim 
        self.inv_sqrt_head_dim = 1/np.sqrt(head_dim)
        self.inv_sqrt_atten_dim = 1/np.sqrt(self.atten_dim)
        self.depth = depth

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(self.atten_dim)
        self.values = []
        self.keys = []
        self.queries = []
        self.convolution = []
        self.roll_out = []

        for _ in range(depth):
            input_sz = self.atten_dim
            self.values.append(nn.Linear(input_sz, self.atten_dim))
            self.keys.append(nn.Linear(input_sz, self.atten_dim))
            self.queries.append(nn.Linear(input_sz, self.atten_dim))
            self.convolution.append(nn.Conv1d(head_dim, self.atten_dim, 1))
            self.roll_out.append(nn.Sequential(
                nn.Linear(self.atten_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.atten_dim),
                nn.ReLU()
            ))
        self.entity_encodings = nn.Sequential(
            nn.Conv1d(self.atten_dim, self.atten_dim, 1),
            nn.ReLU()
        )
        self.spatial_entity_encodings = nn.Sequential(
            nn.Conv1d(1, 1, self.atten_dim//seesz, self.atten_dim//seesz),
            nn.ReLU()
        )
        self.encoded_entities = nn.Sequential(
            nn.Linear(self.atten_dim, self.atten_dim),
            nn.ReLU()
        )
        
    def forward(self, embedded_entities, positional_encoding):
        X = Tensor(np.array(embedded_entities + positional_encoding * self.inv_sqrt_atten_dim))
        n = X.size(0)

        step_input = X
        for i in range(self.depth):
            # pass through Query, Key, Value layers and reorder to: (head_ct, n, head_dim)
            Q = self.queries[i](step_input).view(-1, self.head_ct, self.head_dim).transpose(0, 1)
            K = self.keys[i](step_input).view(-1, self.head_ct, self.head_dim).transpose(0, 1)
            V = self.values[i](step_input).view(-1, self.head_ct, self.head_dim).transpose(0, 1)

            # Query & Key Lane
            QK_similarity = matmul(Q, K.transpose(-2, -1)) * self.inv_sqrt_head_dim
            QK_probs = self.softmax(QK_similarity)

            # Combine with Value
            attention = matmul(QK_probs, V).transpose(1, 2)

            # convolve to double channels and sum across heads
            convolution_sum = tsum(self.convolution[i](attention), 0).transpose(0, 1)

            # It's a transformer!
            roll_out = self.roll_out[i](convolution_sum)

            # Skip connection
            step_input = self.layer_norm(roll_out + step_input)

        transformer_output = step_input[None, :, :]
        entity_encodings = self.entity_encodings(transformer_output.transpose(2, 1)).transpose(2, 1)
        entenc_cnn_prep = entity_encodings.transpose(0, 1)
        spatial_entity_encodings = self.spatial_entity_encodings(entenc_cnn_prep).squeeze()
        encoded_entities = self.encoded_entities(tmean(entity_encodings, 0))

        return entity_encodings.squeeze(), spatial_entity_encodings, encoded_entities


"""
A CNN, the latter half of which is a ResNet.

Inputs:
The map, dimensions (layer_ct x map_dim x map_dim), where layers are formed from data such as 
walkability, visibility, and entity encodings.
Outputs:
A single vector encoding of the map as well as the intermediate skip connections from the ResNet.

The detail of how the map is preprocessed and how entity encodings are added to the map can be found
primarily in processing.py
"""
class SpatialEncoder(nn.Module):

    def __init__(self, map_dim=128, map_depth=36, res_block_ct=4):
        super(SpatialEncoder, self).__init__()

        self.map_depth = map_depth
        self.map_dim = map_dim
        self.res_block_ct = res_block_ct

        self.down_sample = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.res_blocks = [ResModule(
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU()
            )) for _ in range(res_block_ct)
        ]
        self.embedded_spacial = nn.Sequential(
            nn.Conv2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear((128 * 16**2 * map_depth) // 2**3, 256),
            nn.ReLU()
        )

    def forward(self, preprocessed_map):
        _map = Tensor(preprocessed_map)
        flat_map = tflatten(_map, 0, 1)[None, :, :]
        res_block_input = self.down_sample(flat_map)

        map_skips = []
        for i in range(self.res_block_ct):
            res_block_input = self.res_blocks[i](res_block_input)
            map_skips.append(res_block_input)
        res_net_output = res_block_input

        encoded_spatial = self.embedded_spacial(res_net_output)

        return encoded_spatial, map_skips