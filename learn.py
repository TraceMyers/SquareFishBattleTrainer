import numpy as np
from torch import nn, set_num_threads, optim, cuda, Tensor, zeros as tzeros, autograd, sum as tsum, device as tdevice, load as tload, save as tsave, argmax as targmax
from torchvision import transforms
from torchsummary import summary as tsummary
from os import stat, fdopen, dup as osdup
import sys
from struct import unpack
from matplotlib import pyplot as plt
from load import load_entities, load_map, load_orders, load_file
from processing import init_cache, preprocess_entities, prime_embedding
from models import BWNet
from pickle import load as pload, dump as pdump
from random import choices as randchoices
from multiprocessing import Process, Queue

FILE_RANGE_MAX = 474


def get_pickled_file():
    file_data = None
    try:
        with open('data/pickled_data.bin', 'rb') as f:
            file_data = pload(f, encoding='bytes')
    except:
        print(f'learn::pickled_file_testing(): could not read pickled file. returning None')
    return file_data

   
def train(process_queue, save_model=True, load_model=True, pickled_data=False, file_ct=FILE_RANGE_MAX):
    init_cache()
    autograd.set_detect_anomaly(True)
    device = tdevice("cuda")
    bw_net = BWNet()
    if load_model:
        print('loading model')
        bw_net.load_state_dict(tload('models/supervised.mod'))
        with open('models/supervised_progress.txt', 'r') as f:
            line = f.readline()
            vals = line.split(",")
            file_start = int(vals[0])
            frame_start = int(vals[1])
            print(f'starting at file {file_start}, frame {frame_start}')
    else:
        file_start = 0
        frame_start = 0
        print('using new parameters')
    bw_net = bw_net.to(device)
    bw_net.train()
    kldiv_loss = nn.KLDivLoss(reduction='sum')
    mle_loss = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(bw_net.parameters(), betas=(0.9, 0.999), lr=1e-3)

    action_ct = 17
    train_losses = [0]
    target_losses = [0]
    actor_losses = [0]
    location_losses = [0]

    var = 16
    const1 = 1/np.sqrt(2*np.pi*var)
    const2 = -1/(2*var)

    norm_calc_x = tzeros((128, 128)).cuda()
    norm_calc_y = tzeros((128, 128)).cuda()
    norm_calc_tensors = (norm_calc_y, norm_calc_x)

    for i in range(file_start, file_ct):
        if pickled_data:
            # WARNING/TODO: pickled file has bad map data. need to remake
            file_data = get_pickled_file()
        else:
            file_data = load_file(i)
        if file_data is None:
            continue

        bw_net.reset_core()

        entity_frames = file_data[0]
        order_frames = file_data[1]
        walkability = file_data[2][0][None, :, :]
        map_frames = file_data[2][1]
        frame_ct = len(entity_frames)

        for frame_no in range(frame_start, frame_ct):
            if not process_queue.empty():
                print('saving model')
                tsave(bw_net.state_dict(), 'models/supervised.mod')
                with open('models/supervised_progress.txt', 'w') as f:
                    f.write(f'{i},{frame_no}')
                return
            
            # --------------------------------------------------------------------------------------
            # ------------------------------------------------------------------ frame preprocessing
            # --------------------------------------------------------------------------------------

            raw_entities = entity_frames[frame_no]
            self_units_mask = raw_entities[:, 1] == 0 # everything had 1 subtracted in preprocessing
            self_units = raw_entities[self_units_mask]
            self_units_ct = self_units.shape[0]
            preprocessed_entities, positional_encoding, positions = \
                preprocess_entities(raw_entities)
            
            if preprocessed_entities is None:
                continue
            # embedded_entities = prime_embedding(preprocessed_entities)

            raw_orders = order_frames[frame_no]

            # --------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------- train 
            # --------------------------------------------------------------------------------------
            optimizer.zero_grad()
            error_pairs = \
                bw_net.forward(
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
                    (const1, const2),
                    i,
                    norm_calc_tensors
                )
            if error_pairs == None:
                continue
            target_error_pair = error_pairs[0]
            location_error_pair = error_pairs[1]
            actor_error_pair = error_pairs[2]
            target_loss, loc_loss, actor_loss = 0, 0, 0

            if target_error_pair[0] is not None:
                target_loss = \
                    autograd.Variable(
                        kldiv_loss(target_error_pair[0], target_error_pair[1]), 
                        requires_grad=True
                    ) + \
                    autograd.Variable(
                        l2_loss(target_error_pair[0], target_error_pair[1]), 
                        requires_grad=True
                    ) * 1e-5
                location_losses.append(location_losses[-1])
                target_losses.append(target_loss.item())
            elif location_error_pair[0] is not None:
                location_label = targmax(location_error_pair[1])
                loc_loss = \
                    autograd.Variable(
                        mle_loss(location_error_pair[0], location_label), 
                        requires_grad=True
                    ) + \
                    autograd.Variable(
                        l2_loss(location_error_pair[0], location_error_pair[1]), 
                        requires_grad=True
                    ) * 1e-5 + \
                    autograd.Variable(
                        kldiv_loss(location_error_pair[0], location_error_pair[2]), 
                        requires_grad=True
                    )
                location_losses.append(loc_loss.item())
                target_losses.append(target_losses[-1])
            actor_loss = \
                autograd.Variable(
                    kldiv_loss(actor_error_pair[0], actor_error_pair[1]), 
                    requires_grad=True
                ) + \
                autograd.Variable(
                    l2_loss(actor_error_pair[0], actor_error_pair[1]), 
                    requires_grad=True
                ) * 1e-5
            actor_losses.append(actor_loss.item())

            loss = loc_loss + target_loss + actor_loss
            if loss > 0:
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                bw_net.post_backward()
                train_losses.append(loss.item())
                print(location_losses[-1])

        frame_start = 0
        print('saving model')
        tsave(bw_net.state_dict(), 'models/supervised.mod')
        with open('models/supervised_progress.txt', 'w') as f:
            f.write(f'{i+1},0')

if __name__ == '__main__':
    process_queue = Queue()
    # save model, load_model, pickled data
    # train(process_queue, False, False, False)
    train_process = Process(target=train, args=(process_queue, True, True, False))
    train_process.start()
    _input = input()
    process_queue.put(1)
    
    
    