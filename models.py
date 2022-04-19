from torch import nn, set_num_threads, optim, cuda, Tensor, matmul, mean as tmean
from torchvision import transforms
from torchsummary import summary as tsummary
import numpy as np

class EntityEncoder(nn.Module):

    def __init__(self, entity_col_ct, head_ct=2, head_dim=128, depth=3):
        super(EntityEncoder, self).__init__()

        self.cache =  []

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
        self.roll_out = []

        for _ in range(depth):
            input_sz = self.atten_dim
            self.values.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
            ))
            self.keys.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
            ))
            self.queries.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
            ))
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
        self.encoded_entities = nn.Sequential(
            nn.Linear(self.atten_dim, self.atten_dim),
            nn.ReLU()
        )

    def add_to_batch(self, embedded_entities, positional_encoding):    
        self.cache.append(embedded_entities + positional_encoding * self.inv_sqrt_atten_dim)

    def flush_batch(self):
        self.cache.clear()

    def forward(self):
        X = Tensor(np.array(self.cache))
        batch_size = X.size(0)

        step_input = X
        for i in range(self.depth):
            # pass through these layers and reorder to: (batch size, head_ct, n, head_dim)
            Q = self.queries[i](step_input).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)
            K = self.keys[i](step_input).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)
            V = self.values[i](step_input).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)

            # Query & Key Lane
            QK_similarity = matmul(Q, K.transpose(-2, -1)) * self.inv_sqrt_head_dim
            QK_probs = self.softmax(QK_similarity)
            QK_probs[0, 0, :, 1].sum()

            # Combine with Value
            attention = matmul(QK_probs, V)
            attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.atten_dim)

            # It's a transformer
            roll_out = self.roll_out[i](attention)

            # Skip connection
            step_input = self.layer_norm(roll_out + step_input)

        transformer_output = step_input
        entity_encodings = self.entity_encodings(transformer_output.transpose(1, 2)).transpose(1, 2)
        encoded_entities = self.encoded_entities(tmean(entity_encodings, 1))
        print(entity_encodings.size())
        print(encoded_entities.size())

        

        # 1. Nx256 in
        # 2. run each into 1