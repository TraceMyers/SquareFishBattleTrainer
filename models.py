from torch import nn, set_num_threads, optim, cuda, Tensor, matmul, bmm, var
from torchvision import transforms
from torchsummary import summary as tsummary
import numpy as np

class EntityEncoder(nn.Module):

    def __init__(self, entity_col_ct, head_ct=2, head_dim=128, depth=3):
        super(EntityEncoder, self).__init__()

        self.X = []

        self.head_ct = head_ct
        self.head_dim = head_dim 
        self.atten_dim = head_dim*head_ct
        self.inv_sqrt_entity_dim = 1/np.sqrt(entity_col_ct)
        self.inv_sqrt_head_dim = 1/np.sqrt(head_dim)
        self.depth = depth

        self.softmax = nn.Softmax(dim=-1)
        self.values = []
        self.keys = []
        self.queries = []
        self.attention_out = []

        for i in range(depth):
            if i == 0:
                input_sz = entity_col_ct
            else:
                input_sz = self.atten_dim
            self.values.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
                nn.ReLU()
            ))
            self.keys.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
                nn.ReLU()
            ))
            self.queries.append(nn.Sequential(
                nn.Linear(input_sz, self.atten_dim),
                nn.ReLU()
            ))
            self.attention_out.append(nn.Sequential(
                nn.Linear(self.atten_dim, self.atten_dim),
                nn.ReLU()
            ))

    def add_to_batch(self, entities, positions):    
        self.X.append(np.array(entities + positions * self.inv_sqrt_entity_dim))

    def flush_batch(self):
        self.X.clear()

    def forward(self):
        X = Tensor(np.array(self.X))
        batch_size = X.size(0)

        for i in range(self.depth):
            # pass through these layers and reorder to: (batch size, head_ct, n, head_dim)
            Q = self.queries[i](X).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)
            K = self.keys[i](X).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)
            V = self.values[i](X).view(batch_size, -1, self.head_ct, self.head_dim).transpose(1, 2)

            # Query & Key Lane
            QK_similarity = matmul(Q, K.transpose(-2, -1)) * self.inv_sqrt_head_dim
            QK_probs = self.softmax(QK_similarity)
            QK_probs[0, 0, :, 1].sum()
            # print(QK_probs.size())
            # print(V.size())
            attention = matmul(QK_probs, V)
            attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.atten_dim)
            print(attention.size())
            
            quit()
        

        # 1. Nx256 in
        # 2. run each into 1