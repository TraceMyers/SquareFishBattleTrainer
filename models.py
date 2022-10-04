"""
The models in this file were made possible due to the work done by DeepMind on their project 
Alphastar. The models are translated and modified from those models described by DeepMind
int the file 'alphastar_detail.txt'.
"""

from torch import nn, cuda, Tensor, matmul, mean as tmean, sum as tsum, flatten as tflatten, concat, argmax, zeros as tzeros, any as t_any, log as tlog, exp as texp, logical_not, nan_to_num, max as tmax, add as t_add, isnan as t_isnan, sqrt as tsqrt, exp as texp, normal as tnormal, var as tvar
from torchvision import transforms
from torchsummary import summary as tsummary
import numpy as np
from processing import transformer_positional_encoding, preprocess_map, get_supervised_order, get_action_mask, get_unit_type_action_mask, get_unit_selection_mask, get_order_type, stats
from matplotlib import pyplot as plt
from os import getcwd


ACTION_CT = 17


"""
Module with built-in skip connection
"""
class ResModule(nn.Module):

    def __init__(self, module):
        super(ResModule, self).__init__()
        self.module = module

    def forward(self, inputs):
        ip = self.module(inputs)
        skip = inputs
        return ip + skip


"""
The ResBlock was proposed with nonlinear activation at the end of the block, but according to 
https://arxiv.org/pdf/1603.05027.pdf, identity appears to work just as well if not better.
"""
class ResBlock(nn.Module):

    def __init__(self, input_width, hidden_width, output_width):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_width),
            nn.ReLU()
        )
        self.layer_normalize = nn.LayerNorm(output_width)

    def forward(self, inputs):
        return self.layer_normalize(self.block(inputs.clone()) + inputs)

"""
Gated Linear Unit (always uses scalar context values as the gate here)
"""
class GLU(nn.Module):

    def __init__(self, input_width, gate_width, output_width):
        super(GLU, self).__init__()
        self.output_width = output_width
        self.gate = nn.Sequential(
            nn.Linear(gate_width, input_width),
            nn.Sigmoid()
        ).cuda()
        self.values = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.ReLU()
        ).cuda()

    def forward(self, input_data, gate_data):
        return self.values(input_data * self.gate(gate_data))


"""
This one twerked my brain hard.
"""
class GatedFiLMResBlock(nn.Module):

    def __init__(self, channels=128):
        super(GatedFiLMResBlock, self).__init__()
        self.block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels)
            ),
            nn.ReLU()
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, map_data, reshaped_arenc, map_skip, film_betas, film_gammas):
        skip = self.block[0](map_data)
        step = self.block[1](skip[None, :, :, :]).squeeze()
        filmed = step * film_gammas[:, None, None] + film_betas[:, None, None]
        pre_gate = self.block[2](filmed)
        gated = pre_gate * self.gate(reshaped_arenc)
        return self.output(gated) + map_skip + map_data


class GRU(nn.Module):
    def __init__(self, width):
        super(GRU, self).__init__()
        self.reset = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Sigmoid()
        )
        self.update = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Sigmoid()
        )
        self.pre_output = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Tanh()
        )
    
    def forward(self, input_data, state):
        _id = input_data.squeeze()
        _st = state.squeeze()
        concatenated = concat((_id, _st))
        reset = self.reset(concatenated) * state
        update = self.update(concatenated)
        hidden_state = (1 - update) * state
        pre_output = self.pre_output(concat((reset, _id))) * update
        state_output = hidden_state + pre_output
        return state_output


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

        self.entity_embedding = nn.Sequential(
            nn.Linear(stats('entity_col_ct'), self.atten_dim)
        )
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

        self.values = nn.ModuleList(self.values)
        self.keys = nn.ModuleList(self.keys)
        self.queries = nn.ModuleList(self.queries)
        self.convolution = nn.ModuleList(self.convolution)
        self.roll_out = nn.ModuleList(self.roll_out)

    def forward(self, encoded_entities, positional_encoding):
        embedded_entities = self.entity_embedding(Tensor(encoded_entities).cuda())
        X = embedded_entities + Tensor(np.array(positional_encoding * self.inv_sqrt_atten_dim)).cuda()
        n = X.size(0)

        # print('---\n---\nembedded_entities\n---')
        # print(embedded_entities)
        # print('---\n---\npositional_encoding\n---')
        # print(positional_encoding)

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
        entity_encodings = entity_encodings.squeeze()
        encoded_entities = self.encoded_entities(tmean(entity_encodings, 0))

        return entity_encodings, spatial_entity_encodings, encoded_entities


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
            nn.Conv2d(36, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(*[
                ResModule(nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.ReLU()
                )) 
                for _a in range(2)
            ]) 
            for _b in range(res_block_ct)
        ])
        self.embedded_spacial = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, 4, 2, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU()
            )
        ])

    def forward(self, preprocessed_map):
        _map = Tensor(preprocessed_map).cuda()
        map_skips = []
        map_skips.append(self.down_sample(_map))

        for i in range(self.res_block_ct):
            map_skips.append(self.res_blocks[i](map_skips[i - 1]))
        res_net_output = map_skips[-1]

        intermed_1 = self.embedded_spacial[0](res_net_output)
        intermed_2 = tflatten(intermed_1, start_dim=0)
        encoded_spatial = self.embedded_spacial[1](intermed_2)
        
        return encoded_spatial, map_skips[:-1]


class ScalarEncoder(nn.Module):

    def __init__(self, width=64):
        super(ScalarEncoder, self).__init__()

        self.width = width
        self.action_encoding = nn.Sequential(
            nn.Linear(ACTION_CT, width),
            nn.ReLU()
        )

    def forward(self, time_step, action_mask):
        positional_encoding = Tensor(transformer_positional_encoding(time_step, self.width)).cuda()
        scalar_context = self.action_encoding(Tensor(action_mask).cuda())
        encoded_scalars = concat((positional_encoding, scalar_context))

        return scalar_context, encoded_scalars


"""
A typical LSTM with added layer normalization at the gates.
"""
class LSTM(nn.Module):

    def __init__(self, input_width, hidden_width):
        super(LSTM, self).__init__()

        self.forget_gate = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.Sigmoid()
        )
        self.input_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.Tanh()
            )
        ])
        self.output_gate = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_width)
        self.tanh = nn.Tanh()

    def forward(self, throughput_vec, previous_hidden_state):
        
        forget = self.layer_norm(self.forget_gate(throughput_vec.clone()))
        hidden_state = forget * previous_hidden_state

        remember = self.layer_norm(
            self.input_gate[0](throughput_vec.clone()) * self.input_gate[1](throughput_vec.clone())
        )
        new_hidden_state = remember + hidden_state

        output = self.tanh(new_hidden_state) * self.layer_norm(self.output_gate(throughput_vec.clone()))

        return output, new_hidden_state


"""
A deep LSTM with 3 hidden layers, no projection.

The core of the greater network - the first place in the network where all of the input
is combined.
"""
class Core(nn.Module):

    """
    @input width:
        encoded_entity: 256
        encoded_spatial: 256
        encoded_time: 64
    """
    def __init__(self, input_width=576, hidden_width=384, depth=3):
        super(Core, self).__init__()

        self.lstm_layers = nn.ModuleList([
            LSTM(input_width + hidden_width, hidden_width),
            LSTM(hidden_width, hidden_width),
            LSTM(hidden_width, hidden_width)
        ])
        self.depth = depth

    def forward(
        self, 
        encoded_entities, 
        encoded_spatial, 
        encoded_time, 
        previous_output, 
        previous_hidden_state
    ):
        enc_ent = encoded_entities
        enc_spa = encoded_spatial
        enc_tim = encoded_time
        pre_out = previous_output
        throughput = concat((enc_ent, enc_spa, enc_tim, pre_out))
        hidden_state = previous_hidden_state

        for i in range(len(self.lstm_layers)):
            throughput, hidden_state = self.lstm_layers[i](throughput, hidden_state)

        return throughput, hidden_state


class ActionHead(nn.Module):

    def __init__(self, core_output_width=384, resblock_width=256, resnet_len=12, gate_sz=64):
        super(ActionHead, self).__init__()
        
        core_encoder = [
            nn.Linear(core_output_width, resblock_width),
            nn.LeakyReLU()
        ]
        core_encoder.extend([
            ResBlock(resblock_width, resblock_width, resblock_width) for _ in range(resnet_len)
        ])
        core_encoder.append(nn.ReLU())
        self.action = nn.Sequential(*core_encoder)
        self.action_gate = GLU(resblock_width, gate_sz, ACTION_CT)
        
        self.autoregressive_encoding = [nn.Sequential(
            nn.ReLU(),
            nn.Linear(ACTION_CT, 256),
            nn.ReLU()
        )]
        self.autoregressive_encoding.append(nn.Sequential(
            nn.Linear(core_output_width, 1024),
            nn.ReLU()
        ))
        self.autoregressive_encoding = nn.ModuleList(self.autoregressive_encoding)
        self.ae_gates = [
            GLU(256, gate_sz, 1024),
            GLU(1024, gate_sz, 1024)
        ]

    def forward(self, core_output, scalar_context, supervised_action=None):

        if supervised_action is None:
            # for reinforcement learning
            action_intermediate = self.action(core_output)    
            gated_action = self.action_gate(action_intermediate, scalar_context)

            # TODO: sample from decision space rather than picking argmax to keep decision variance
            action = tzeros(gated_action.shape[0]).cuda()
            action[argmax(gated_action)] = 1
        else:
            # taking the ground truth action and figuring out what to do with it in other parts
            gated_action = None
            action = supervised_action
        
        auto_re_0 = self.autoregressive_encoding[0](action)
        gated_auto_re_0 = self.ae_gates[0](auto_re_0, scalar_context)
        auto_re_1 = self.autoregressive_encoding[1](core_output)
        gated_auto_re_1 = self.ae_gates[1](auto_re_1, scalar_context)
        autoregressive_encoding = gated_auto_re_0 + gated_auto_re_1

        return action, gated_action, autoregressive_encoding
        

class ActorsHead(nn.Module):

    def __init__(self):
        super(ActorsHead, self).__init__()

        self.func_embed = nn.Sequential(
            nn.Linear(233, 256),
            nn.ReLU()
        ).cuda()
        self.entity_keys = nn.Conv1d(256, 32, 1)
        self.argenc = nn.ModuleList([
            nn.Linear(1024, 256),
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU()
            ),
            LSTM(32*2, 32),
            nn.Sequential(
                nn.Linear(32, 1024),
                nn.ReLU()
            )
        ])
        self.sigmoid = nn.Sigmoid()

    # thanks to chasep255 on stackoverflow
    def temp_softmax(self, logit, temp=0.8):
        vec = texp(tlog(logit) / temp)
        vec = nan_to_num(vec)
        _sum = tsum(vec)
        if _sum != 0:
            vec = vec / _sum
            return vec
        return None

    @staticmethod
    def sample(vec):
        k = np.argmax(np.random.multinomial(1, vec.cpu().detach().numpy().squeeze(), 1))
        return k
    
    def forward(self, utype_mask, entity_mask, entity_encodings, autoregressive_encoding, self_unit_ct):
        func_embed = self.func_embed(Tensor(utype_mask).cuda())
        entity_mask = Tensor(entity_mask).cuda()
        entity_keys = self.entity_keys(entity_encodings[None, :, :].transpose(1, 2))
        entity_keys = entity_keys.transpose(1, 2).squeeze().transpose(0, 1)
        entity_ct = entity_mask.shape[0]
        selected_units = tzeros(entity_ct).cuda()
        unit_logits = tzeros((entity_ct, entity_ct)).cuda()
        hidden_lstm_state = tzeros(32).cuda()
        query = tzeros(32).cuda()

        for ent in range(min(64, self_unit_ct)):
            intermed0 = t_add(self.argenc[0](autoregressive_encoding.clone().cuda()), func_embed).cuda()
            intermed1 = self.argenc[1](intermed0)
            query, hidden_lstm_state = self.argenc[2](concat((intermed1, query)), hidden_lstm_state)

            similarity = matmul(query[None, :], entity_keys)

            soft = self.temp_softmax(self.sigmoid(similarity))
            if soft == None:
                continue
            unit_logits[ent, :] = soft

            sample_pick = self.sample(soft)
            one_hot = tzeros(entity_ct).cuda()
            one_hot[sample_pick] = 1
            if not t_any(one_hot * entity_mask):
                continue
            
            entity_mask[sample_pick] = 0
            selected_units[sample_pick] = 1

            selection = matmul(one_hot[None, :], entity_keys.transpose(0,1))
            selection = (selection - tmean(selection)).squeeze()
            if tsum(t_isnan(selection)) == 0:
                autoregressive_encoding += self.argenc[3](selection)
            else:
                break

        return unit_logits, selected_units, autoregressive_encoding
        

class TargetHead(nn.Module):

    def __init__(self):
        super(TargetHead, self).__init__()

        self.func_embed = nn.Sequential(
            nn.Linear(233, 256),
            nn.ReLU()
        )
        self.entity_keys = nn.Conv1d(256, 32, 1)
        self.argenc = [
            nn.Linear(1024, 256),
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU()
            ),
            LSTM(32*2, 32)
        ]
        self.sigmoid = nn.Sigmoid()

    # thanks to chasep255 on stackoverflow
    def temp_softmax(self, logit, temp=0.8):
        vec = texp(tlog(logit) / temp)
        vec = nan_to_num(vec)
        _sum = tsum(vec)
        if _sum != 0:
            vec = vec / _sum
        return vec

    @staticmethod
    def sample(vec):
        return np.argmax(np.random.multinomial(1, vec.cpu().detach().numpy().squeeze(), 1))
    
    def forward(self, utype_mask, entity_mask, entity_encodings, autoregressive_encoding, self_unit_ct):
        # func_embed = self.func_embed(Tensor(utype_mask))
        entity_mask = logical_not(Tensor(entity_mask).cuda()) * 1.0
        entity_keys = self.entity_keys(entity_encodings[None, :, :].transpose(1, 2))
        entity_keys = entity_keys.transpose(1, 2).squeeze().transpose(0, 1)
        entity_ct = entity_mask.shape[0]
        hidden_lstm_state = tzeros(32).cuda()
        query = tzeros(32).cuda()
        targeted_unit = tzeros(entity_ct).cuda()


        if self_unit_ct - entity_encodings.size(0) > 0:
            intermed = self.argenc[0](autoregressive_encoding) # + func_embed
            intermed = self.argenc[1](intermed)
            query, hidden_lstm_state = self.argenc[2](concat((intermed, query)), hidden_lstm_state)

            similarity = matmul(query[None, :], entity_keys)

            unit_logits = self.temp_softmax(self.sigmoid(similarity))

            sample_pick = self.sample(unit_logits.clone())
            one_hot = tzeros(entity_ct).cuda()
            one_hot[sample_pick] = 1

            if t_any(one_hot * entity_mask):
                entity_mask[sample_pick] = 0
                targeted_unit[sample_pick] = 1

        return unit_logits, targeted_unit
        
"""
Here is where I realized that I concatenated map layers in a silly way. Will it still work?
"""
class LocationHead(nn.Module):

    def __init__(self, map_channels=128, map_skip_wh=16, map_skip_ct=4):
        super(LocationHead, self).__init__()

        self.gru_width = map_channels*2

        self.map_skip_ct = map_skip_ct
        self.map_channels = map_channels

        self.arenc_map_convolution = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.map_channels + 4, 128, 1),
            nn.ReLU()
        )
        self.gated_film_resblocks = nn.ModuleList([
            GatedFiLMResBlock() for _ in range(map_skip_ct)
        ])
        self.map_skip_gru_prep = nn.Sequential(
            nn.Conv2d(map_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.ReLU()
        )
        self.autoenc_grus = nn.ModuleList([GRU(self.gru_width) for _ in range(map_skip_ct)])
        self.map_skip_grus = nn.ModuleList([GRU(self.gru_width) for _ in range(map_skip_ct)])
        self.film_params = nn.Linear(2*self.gru_width, 2*map_channels*map_skip_ct)
        self.final_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, 2),
            nn.ReLU()
        )

    def forward(self, autoregressive_encoding, walkability_mask, map_skips, _iter, file_no):
        autoregressive_projection = autoregressive_encoding.view(4, 16, 16)
        autoregressive_projection_2 = autoregressive_projection.view(4, 256)
        arenc_map_concat = concat((autoregressive_projection, map_skips[-1]))
        autoenc_gru_state = tzeros(self.gru_width, requires_grad=True).cuda()
        map_skip_gru_state = tzeros(self.gru_width, requires_grad=True).cuda()

        for i in range(self.map_skip_ct):
            autoenc_gru_state = self.autoenc_grus[i](autoregressive_projection_2[i], autoenc_gru_state)
            prepped_skip = self.map_skip_gru_prep(map_skips[i]).view(256)
            map_skip_gru_state = self.map_skip_grus[i](prepped_skip, map_skip_gru_state)
        film_params = self.film_params(
            concat((autoenc_gru_state, map_skip_gru_state))
        ).view(self.map_skip_ct, 2*self.map_channels)

        throughput = self.arenc_map_convolution(arenc_map_concat)
        for i in range(self.map_skip_ct):
            throughput = self.gated_film_resblocks[i](
                throughput, 
                autoregressive_projection,
                map_skips[i],
                film_params[i, :self.map_channels],
                film_params[i, self.map_channels:]
            )
        soft_prediction = (self.final_cnn(throughput).squeeze()) * walkability_mask
        # print(tmax(soft_prediction))
        # quit()
        if _iter % 50 == 0:
            plt.imshow(soft_prediction.cpu().detach().numpy())
            plt.savefig(f'{getcwd()}/imgs/loclearn/{file_no}_{_iter}.jpg')
        max_locs = (soft_prediction == tmax(soft_prediction)).nonzero()
        if max_locs.size(0) == 0:
            location = None
        else:
            location = max_locs[0]
        return soft_prediction, location


class BWNet(nn.Module):

    def __init__(self):
        super(BWNet, self).__init__()
        self.entity_encoder = EntityEncoder()
        self.spatial_encoder = SpatialEncoder()
        self.scalar_encoder = ScalarEncoder()
        self.network_core = Core()
        self.action_head = ActionHead()
        self.actors_head = ActorsHead()
        self.target_head = TargetHead()
        self.location_head = LocationHead()
        self.core_output = tzeros(384, requires_grad=True).cuda()
        # self.core_hidden_state = tzeros(384, requires_grad=True).cuda()
        self.core_hidden_state = tnormal(0, 1, (1, 384), requires_grad=True).squeeze().cuda()

    def reset_core(self):
        self.core_output = tzeros(384, requires_grad=True).cuda()
        # self.core_hidden_state = tzeros(384, requires_grad=True).cuda()
        self.core_hidden_state = tnormal(0, 1, (1, 384), requires_grad=True).squeeze().cuda()

    def forward(
        self, 
        raw_entities,
        preprocessed_entities,
        raw_orders,
        positional_encoding,
        walkability,
        map_frames,
        frame_no,
        positions,
        self_units_ct,
        action_ct,
        norm_consts,
        file_no,
        nct
    ):
        error_pairs = []

        entity_encodings, spatial_entity_encodings, encoded_entities = \
            self.entity_encoder.forward(preprocessed_entities, positional_encoding)

        raw_map = np.append(walkability, map_frames[frame_no], axis=0)
        preprocessed_map, walkability_mask = preprocess_map(
            raw_map, 
            spatial_entity_encodings.cpu().detach().numpy(), 
            positions
        )
        walkability_mask = Tensor(walkability_mask).cuda()

        # 0 order_one_hot, 1 actor_one_hot, 2 target_one_hot, 3 tilepos_one_hot, 4 order_type
        supervised_order = get_supervised_order(raw_orders, self_units_ct, action_ct)
        if supervised_order[4] == -1:
            return None

        for i in range(len(supervised_order)):
            supervised_order[i] = Tensor(supervised_order[i]).cuda()
        action_mask = get_action_mask(raw_entities, action_ct)

        encoded_spatial, map_skips = \
            self.spatial_encoder.forward(preprocessed_map)
        scalar_context, encoded_scalars = \
            self.scalar_encoder.forward(frame_no, action_mask)

        self.core_output, self.core_hidden_state = \
            self.network_core.forward(
                encoded_entities, 
                encoded_spatial, 
                scalar_context,
                self.core_output,
                self.core_hidden_state
            )
        
        one_hot_action, gated_action, autoregressive_encoding = \
            self.action_head.forward(self.core_output, scalar_context, supervised_order[0])

        unit_type_action_mask = get_unit_type_action_mask(one_hot_action.cpu().detach().numpy())
        unit_selection_mask = get_unit_selection_mask(raw_entities.shape[0], self_units_ct)

        unit_logits, selected_units, autoregressive_encoding = \
            self.actors_head.forward(
                unit_type_action_mask,
                unit_selection_mask,
                entity_encodings,
                autoregressive_encoding,
                self_units_ct
            )
        
        order_type = get_order_type(one_hot_action)
        if order_type == 0:
            error_pairs.append((None, None))
            error_pairs.append((None, None))
        elif order_type == 1:
            target_logits, targeted_unit = self.target_head.forward(
                unit_type_action_mask,
                unit_selection_mask,
                entity_encodings,
                autoregressive_encoding,
                self_units_ct
            )

            target_logit_error_tensor = nn.functional.log_softmax(target_logits)
            true_targets = nn.functional.softmax(supervised_order[2])
            error_pairs.append((target_logit_error_tensor, true_targets))
            error_pairs.append((None, None))
        elif order_type == 2:
            error_pairs.append((None, None))
            softpred_map, location = self.location_head.forward(
                autoregressive_encoding, 
                walkability_mask, 
                map_skips,
                frame_no,
                file_no
            )
            location_logit = nn.functional.log_softmax(tflatten(softpred_map.clone()))
            true_location_map = supervised_order[3]
            true_loc = supervised_order[5]

            ysq_dist = (nct[0] - true_loc[0])**2
            xsq_dist = (nct[1] - true_loc[1])**2
            norm_location_map = norm_consts[0] * texp(norm_consts[1] * (ysq_dist + xsq_dist))
            true_location_map = tflatten(true_location_map)
            norm_location_map = nn.functional.softmax(tflatten(norm_location_map * walkability_mask))
            error_pairs.append((location_logit, true_location_map, norm_location_map))

        unit_logit_maxes = tmax(unit_logits, dim=0)[0]
        unit_logit_means = tmean(unit_logits, dim=0)
        unselected_units = logical_not(selected_units)
        
        u1 = \
            unit_logit_maxes * selected_units + \
            unit_logit_means * unselected_units
        unit_logit_error_tensor = nn.functional.log_softmax(u1)
        true_units = nn.functional.softmax(supervised_order[1])

        error_pairs.append((unit_logit_error_tensor, true_units))

        return error_pairs

    def post_backward(self):
        self.core_output = self.core_output.cpu().detach().requires_grad_(True).cuda()
        self.core_hidden_state = self.core_hidden_state.cpu().detach().requires_grad_(True).cuda()
