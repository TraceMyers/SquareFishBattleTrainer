"""
The models in this file were made possible due to the work done by DeepMind on their project 
Alphastar. The models are largely translated and modified from those models described by DeepMind.
"""

from torch import nn, cuda, Tensor, matmul, mean as tmean, sum as tsum
from torchvision import transforms
from torchsummary import summary as tsummary
import numpy as np


"""
Where Alphastar describes units with concatenated one-hots for attributes plus a concatenated 
binary-encoded position, KochiOno does a simple prime number-based embedding of the concatenated 
attribute one-hots to conform the size of the attribute vector to the size of the encoder. This
simplifies skip-connections after attention heads. It also uses the kind of positional encoding
typical of Transformers, but adapted for 2d (x, y) input. The implementations of those encoders
are in processing.py.
"""
class EntityEncoder(nn.Module):

    def __init__(self, head_ct=2, head_dim=128, depth=3):
        super(EntityEncoder, self).__init__()

        self.head_ct = head_ct
        self.head_dim = head_dim 
        self.atten_dim = head_dim*head_ct
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
            nn.Conv1d(1, 1, self.atten_dim//8, self.atten_dim//8),
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
        print(entity_encodings.size(), spatial_entity_encodings.size(), encoded_entities.size())

        return entity_encodings.squeeze(), spatial_entity_encodings, encoded_entities


class SpatialEncoder(nn.Module):

    def __init__(self):
        super(SpatialEncoder, self).__init__()