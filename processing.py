from tkinter import W
import numpy as np
from matplotlib import pyplot as plt
from random import randint, sample as randsample, choice
from load import load_entities, load_map, load_file
from math import ceil
from sympy import primerange
from pickle import load as pload, dump as pdump
from torch import sum as tsum
import cv2


# --------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------- Timestep ---#
# --------------------------------------------------------------------------------------------------#


"""
For now, time step is the only scalar feature, so the only scalar encoding needed is the positional
encoding of the timestep.
"""
def transformer_positional_encoding(time_step, depth):
    vector = [0 for i in range(depth)]
    for k in range(depth // 2):
        step_val = time_step / 10000**(2 * k / depth)
        vector[2*k] = np.sin(step_val)
        vector[2*k+1] = np.cos(step_val)
    return np.array(vector)


# --------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------- Entities ---#
# --------------------------------------------------------------------------------------------------#


"""
Adapts the original Transformer positional encoding for fairly large (8192x8192) 2d positional 
input. Tested to balance complexity of function and representing nearby x,y points with near 
positional vectors, and far-apart x,y points with far positional vectors; the hope is that the 
function doesn't have to be precisely parametrically learned by the model, since distance is a 
valuable metric.
"""
def positional_encoding_vector(x, y, depth, w, h):
    # skipping very low minimum manhattan distance between vectors in small x, y to avoid euclidian
    # distance going to 0 due to insufficient float precision
    skip_distance = 500
    x += skip_distance
    y += skip_distance
    vector = [0 for i in range(depth)]
    max_denom = ((depth // 2)**2 + (w + skip_distance)**2 + (h + skip_distance)**2)**(1/4)
    for k in range(depth // 2):
        step_val = (y + k) / 10000**(((k+1)**2 + x**2 + y**2)**(1/4) / max_denom)
        vector[2*k] = np.sin(step_val)
        vector[2*k+1] = np.cos(step_val)
    return vector


"""
Creates an array of positional encoding vectors from 2d input using positional_encoding_vector()
    @positions: list of 2d positions; variance is well preserved in the < 16,000 range
    @depth: depth (length) of the positional encoding vectors
    @w: width of the space (max x)
    @h: height of the space (max y)
"""
def positional_encoding_array(positions, depth, w, h):
    array = [None for _ in range(len(positions))]
    for i in range(len(positions)):
        position = positions[i]
        pos_tuple = tuple(position)
        cached_vec = _cache.pem_cached(pos_tuple)
        if cached_vec == None:
            vec = positional_encoding_vector(position[0], position[1], depth, w, h)
            array[i] = vec
            _cache.pem_add(pos_tuple, vec)
        else:
            array[i] = cached_vec
    return np.array(array)


"""
'euclidian' setting may render minimum distance == 0.0f
"""
def positional_encoding_array_distances(array, dist_type='manhattan'):
    assert dist_type in ['euclidian', 'manhattan']
    d = array.shape[0]
    dist_mat = np.zeros((d, d))
    _sum = 0
    _min = 1000000.0
    n = 0
    for i in range(d):
        for j in range(i+1, d):
            if i != j:
                if dist_type == 'euclidian':
                    dist = np.sum((array[i,:] - array[j,:])**2)
                else:
                    dist = np.sum(np.abs(array[i,:] - array[j,:]))
                _sum += dist
                if dist < _min:
                    _min = dist
                dist_mat[i, j] = dist
                n += 1
    return dist_mat, _sum/n, np.max(dist_mat), _min


"""
Used for tuning the encoder so that it has sufficient minimum distance between vectors, so that the
vectors near in x, y have shorter distance, and to balance those outcomes with simplification of
the encoding algorithm for easier learning.
"""
def test_positional_encoding_array(
        test_type='visual', 
        depth=256, 
        width=8192, 
        height=8192, 
        xy_plane_ct=8
    ):
    assert test_type in ['visual', 'distance', 'both']
    assert xy_plane_ct >= 4
    xy_plane_dim = 30
    # always including the 4 corners of the xy space
    y_extremes = (0, height - xy_plane_dim, 0, height - xy_plane_dim)
    x_extremes = (0, width - xy_plane_dim, width - xy_plane_dim, 0)
    # the remaining planes are randomly selected
    yrange_starts = [
        randint(xy_plane_dim, y_extremes[1] - xy_plane_dim) for _ in range(xy_plane_ct - 4)
    ]
    xrange_starts = [
        randint(xy_plane_dim, x_extremes[1] - xy_plane_dim) for _ in range(xy_plane_ct - 4)
    ]
    yrange_starts.extend(y_extremes)
    xrange_starts.extend(x_extremes)
    yranges = [range(yrange_starts[i], yrange_starts[i] + xy_plane_dim) for i in range(xy_plane_ct)]
    xranges = [range(xrange_starts[i], xrange_starts[i] + xy_plane_dim) for i in range(xy_plane_ct)]
    xy_planes = [[[(x, y) for x in xranges[i]] for y in yranges[i]] for i in range(xy_plane_ct)]
    
    arrs = []
    for xy_plane in xy_planes:
        y_fixed_plane = []
        for row in xy_plane:
            pos_enc_arr = positional_encoding_array(row, depth, width, height)
            y_fixed_plane.append(pos_enc_arr)
        arrs.append(y_fixed_plane)
    
    arrs = np.array(arrs)
    if test_type == 'visual' or test_type == 'both':
        for i in range(arrs.shape[0]):
            for j in range(arrs.shape[1]):
                arr = arrs[i, j, :, :]
                plt.imshow(arr, cmap='plasma')
                plt.show(block=False)
                plt.pause(0.4)
                plt.close()
                # plt.show()
    elif test_type == 'distance' or test_type == 'both':
        mean_near_xy_dist, max_near_xy_dist, min_near_xy_dist = 0.0, 0.0, 1000000.0
        mean_rand_xy_dist, max_rand_xy_dist, min_rand_xy_dist = 0.0, 0.0, 1000000.0
        # getting distance between positional vectors conditioning on each x, y plane
        for i in range(xy_plane_ct):
            encoded_3d_space = arrs[i, :, :, :]
            for j in range(xy_plane_dim):
                # considering 5 neighbors at a time
                for k in range(0, xy_plane_dim-5, 5):
                    x_fixed = encoded_3d_space[j, k:k+5, :]
                    _, iter_mean_near_xy_dist, iter_max_near_xy_dist, iter_min_near_xy_dist = \
                        positional_encoding_array_distances(x_fixed)
                    mean_near_xy_dist += iter_mean_near_xy_dist
                    if iter_min_near_xy_dist < min_near_xy_dist:
                        min_near_xy_dist = iter_min_near_xy_dist
                    elif iter_max_near_xy_dist > max_near_xy_dist:
                        max_near_xy_dist = iter_max_near_xy_dist
            for j in range(xy_plane_dim):
                # considering 5 neighbors at a time
                for k in range(0, xy_plane_dim-5, 5):
                    y_fixed = encoded_3d_space[k:k+5, j, :]
                    _, iter_mean_near_xy_dist, iter_max_near_xy_dist, iter_min_near_xy_dist = \
                        positional_encoding_array_distances(y_fixed)
                    mean_near_xy_dist += iter_mean_near_xy_dist
                    if iter_min_near_xy_dist < min_near_xy_dist:
                        min_near_xy_dist = iter_min_near_xy_dist
                    elif iter_max_near_xy_dist > max_near_xy_dist:
                        max_near_xy_dist = iter_max_near_xy_dist
        # fixing x and y for comparison across x, y planes 
        for i in range(xy_plane_dim):
            for j in range(xy_plane_dim):
                encoded_3d_space = arrs[:, i, j, :]
                _, iter_mean_rand_xy_dist, iter_max_rand_xy_dist, iter_min_rand_xy_dist = \
                    positional_encoding_array_distances(encoded_3d_space)
                mean_rand_xy_dist += iter_mean_rand_xy_dist
                if iter_min_rand_xy_dist < min_rand_xy_dist:
                    min_rand_xy_dist = iter_min_rand_xy_dist
                elif iter_max_rand_xy_dist > max_rand_xy_dist:
                    max_rand_xy_dist = iter_max_rand_xy_dist

        # dividing all values by depth so comparisons across different depths can be made
        mean_near_xy_dist /= 2 * xy_plane_dim * xy_plane_ct * depth
        mean_rand_xy_dist /= xy_plane_dim**2 * depth
        min_near_xy_dist /= depth
        max_near_xy_dist /= depth
        min_rand_xy_dist /= depth
        max_rand_xy_dist /= depth
        print(':-----------------------------------:')
        print(':2d positional encoder distance test:')
        print(':-----------------------------------:')
        print('Near (x, y) values:')
        print(
            f'\tmin: {min_near_xy_dist:.9f}, max: {max_near_xy_dist:.9f}, ' \
            f'mean: {mean_near_xy_dist:.9f}'
        )
        print('Random (x, y) values:')
        print(
            f'\tmin: {min_rand_xy_dist:.9f}, max: {max_rand_xy_dist:.9f}, ' \
            f'mean: {mean_rand_xy_dist:.9f}'
        )


"""
 0: unit type      (int -> one-hot max 233)
 1: owner          (int -> binary one-hot) 
 2: health         (int -> one hot of sqrt(min(hp, max_bw_hp))) (max bw hp = 2500)
 3: energy         (int -> one hot of sqrt(min(en, max_bw_en))) (max bw en = 250)
 4: attack cd      (int -> one hot with maximum 50 (1 learn interval per 2 frames))
 5: spell cd       (int -> one hot with maximum 50 (1 learn interval per 2 frames))
 6: is burrowed    (int -> binary one-hot)
 7: armor          (int -> one hot maximum 10)
 8: position x     (set aside for positional encoding)
 9: position y     (set aside for positional encoding)   
"""
def preprocess_entities(entities, nn_width=256):
    n = entities.shape[0]
    positions = entities[:, _cache.entity_col_ct-2:]
    entities = entities[:, 0:_cache.entity_col_ct-2]
    entities[:, 2] = (np.sqrt(entities[:, 2].astype(np.float32))).astype(np.int32)
    entities[:, 3] = (np.sqrt(entities[:, 3].astype(np.float32))).astype(np.int32)
    entities[:, 4] = entities[:, 4] // 2
    entities[:, 5] = entities[:, 5] // 2
    entities[1:] -= 1

    preprocessed = np.zeros((n, _cache.preprocessed_entity_col_ct), dtype=np.int32)
    for i in range(n):
        entity = entities[i, 0:_cache.entity_col_ct-2]
        for j in range(len(_cache.one_hot_spacing)):
            entity_val = entity[j]
            if entity_val < 0:
                continue
            spacing = _cache.one_hot_spacing_cumulative[j]
            try:
                preprocessed[i, spacing + entity_val] = 1
            except:
                return None, None, None
    return preprocessed, positional_encoding_array(positions, nn_width, 8192, 8192), positions


"""
An attempt at creating my own static embedding using prime numbers to ensure
uniqueness for each entity permutation. Assumes entity data is preprocessed into
concatenated one-hot vectors. Appears to work alright!
"""
def prime_embedding(preprocessed_entities, encoder_layer_sz=256, show_plots=False):
    in_cols = preprocessed_entities.shape[1]
    n = preprocessed_entities.shape[0]
    step_size = ceil(in_cols / encoder_layer_sz)
    init_step_size = step_size
    embedded_entities = np.zeros((n, encoder_layer_sz))
    primes = _cache.primes[1: in_cols+1]
    min_prime = primes[0]
    div_val = 1/(min_prime*0.9)

    i = 0
    j = 0
    remaining_embedded_cols = encoder_layer_sz
    remaining_preprocessed_cols = in_cols
    while True:
        if i >= in_cols:
            break
        if step_size > 1:
            captured_vals = preprocessed_entities[:,i:i+step_size] * primes[i:i+step_size] * div_val
            embedded_entities[:,j] = np.prod(captured_vals, axis=1, where=(captured_vals > 0))
            if remaining_embedded_cols * (step_size-1) >= remaining_preprocessed_cols:
                step_size -= 1
        else:
            embedded_entities[:,j] = preprocessed_entities[:,i] * primes[i] * div_val
        i += step_size
        j += 1
        remaining_embedded_cols -= 1
        remaining_preprocessed_cols -= step_size

    embedded_entities = np.log(embedded_entities, where=(embedded_entities>0))
    embedded_max = np.log(np.prod(primes[-init_step_size:]))
    embedded_entities /= embedded_max

    if show_plots:
        flattened = embedded_entities.flatten()
        plt.title('embedded nonzero')
        plt.hist(flattened[flattened > 0], bins=100)
        plt.show()
        plt.title('preprocessed')
        plt.imshow(preprocessed_entities, cmap='plasma')
        plt.show()
        plt.title('embedded')
        plt.imshow(embedded_entities, cmap='plasma')
        plt.show()

    return embedded_entities


# --------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------- Orders -----#
# --------------------------------------------------------------------------------------------------#

# considered types:
# 35 - larva
# 37 - zergling 
# 38 - hydralisk
# 39 - ultralisk
# 41 - drone
# 42 - overlord
# 43 - mutalisk
# 44 - guardian
# 45 - queen
# 46 - defiler
# 47 - scourge
# 62 - devourer
# 103 - lurker
def get_action_mask(raw_entities, ACTION_CT):
    self_units_mask = raw_entities[:, 1] == -1 # everything had 1 subtracted in preprocessing
    self_units = raw_entities[self_units_mask] + 1
    types = set(self_units[:, 0])
    other_units_visible = raw_entities.shape[0] > self_units.shape[0]

    action_mask = np.ones(ACTION_CT)
    action_mask[11:] = 0
    action_mask[1] = other_units_visible
    
    attack_types = {
        37, 38, 39, 41, 43, 44, 47, 62
    }
    attacking_possible = False
    for utype in types:
        if utype in attack_types:
            attacking_possible = True
            break
    if not attacking_possible:
        action_mask[0] = 0
        action_mask[1] = 0
    
    if 103 in types:
        lurkers_mask = self_units[:, 0] == 103
        lurkers = self_units[lurkers_mask]
        burrowed_lurker_ct = np.sum(lurkers[:, 6])
        if burrowed_lurker_ct == lurkers.shape[0]:
            action_mask[2] = 0
        elif burrowed_lurker_ct == 0:
            action_mask[9] = 0
    else:
        action_mask[2] = 0
        action_mask[9] = 0

    if 46 in types:
        defilers_mask = self_units[:, 0] == 46
        defilers = self_units[defilers_mask]
        defiler_ct = defilers.shape[0]
        cooling_down_ct = np.sum(defilers[:,5] > 0)
        if cooling_down_ct < defiler_ct:
            plague_energy_ct = np.sum(defilers[:,3] >= 12)
            if plague_energy_ct == 0:
                action_mask[5] = 0
            swarm_energy_ct = np.sum(defilers[:,3] >= 10)
            if swarm_energy_ct == 0:
                action_mask[4] = 0
        else:
            action_mask[3] = 0
    else:
        action_mask[3] = 0
        action_mask[4] = 0
        action_mask[5] = 0
    
    return action_mask


def get_unit_type_action_mask(action_one_hot):
    action_one_hot = action_one_hot.astype(np.int32)
    if np.sum(action_one_hot) == 0:
        return _cache.none_mask
    if np.any(action_one_hot & _cache.attack_move) or np.any(action_one_hot & _cache.attack_unit):
        return _cache.attack_mask
    if np.any(action_one_hot & _cache.burrowing) or np.any(action_one_hot & _cache.unburrowing):
        return _cache.burrow_mask
    if np.any(action_one_hot & _cache.defiler_combined):
        return _cache.defiler_mask
    return _cache.other_mask


def get_unit_selection_mask(unit_ct, self_unit_ct):
    return np.array([1 if i < self_unit_ct else 0 for i in range(unit_ct)])


order_dict = {
    'attack_move': (14, 0), 'attack_unit': (10, 1), 'burrowing': (116, 2),
    'consume': (145, 3), 'dark_swarm': (119, 4), 'plague': (144, 5),
    'hold_position': (107, 6), 'move': (6, 7), 'patrol': (152, 8),
    'unburrowing': (118, 9), 'stop': (1, 10), 'irradiate': (143, 11),
    'emp': (122, 12), 'defense_matrix': (141, 13), 'unsieging': (99, 14),
    'sieging': (98, 15), 'mine': (20, 16)
}
independent_orders = [
    order_dict['burrowing'][1],
    order_dict['hold_position'][1],
    order_dict['unburrowing'][1],
    order_dict['stop'][1],
    order_dict['unsieging'][1],
    order_dict['sieging'][1]
]
positional_orders = [
    order_dict['attack_move'][1],
    order_dict['dark_swarm'][1],
    order_dict['emp'][1],
    order_dict['move'][1],
    order_dict['patrol'][1],
    order_dict['mine'][1],
]
targeted_orders = [
    order_dict['attack_unit'][1],
    order_dict['consume'][1],
    order_dict['defense_matrix'][1],
    order_dict['irradiate'][1],
    order_dict['plague'][1]
]


def get_order_type(order_one_hot):
    order = np.argmax(order_one_hot.cpu().detach().numpy())
    if order in independent_orders:
        return 0
    if order in targeted_orders:
        return 1
    if order in positional_orders:
        return 2
    else:
        return -1
    

def get_supervised_order(raw_orders, self_units_ct, ACTION_CT, walkability_mask_dim=128):
    self_orders = raw_orders[:, :self_units_ct]
    new_orders_mask = self_orders[:, 0] != 0
    new_orders = self_orders[new_orders_mask]

    order_type = -1
    order_one_hot = np.zeros(ACTION_CT)
    actor_one_hot = np.zeros(self_orders.shape[0])
    target_one_hot = np.zeros(raw_orders.shape[0])
    tilepos_one_hot = np.zeros((walkability_mask_dim, walkability_mask_dim))
    order_encodings = [
        order_one_hot,
        actor_one_hot,
        target_one_hot,
        tilepos_one_hot,
        order_type
    ]

    if new_orders.shape[0] == 0:
        return order_encodings
    order_col = self_orders[:, 0]
    for val in order_dict.values():
        bwapi_val = val[0]
        model_val = val[1]
        mask = order_col == bwapi_val
        self_orders[mask, 0] = model_val

    unique, counts = np.unique(order_col[order_col != 0], return_counts=True)
    if len(counts) == 0:
        return order_encodings
    selected_order = unique[np.argmax(counts)]
    avg_scaled_tilepos_y = 0
    avg_scaled_tilepos_x = 0
    # TODO: debug:     
    # > order_one_hot[selected_order] = 1
    # > IndexError: index 189 is out of bounds for axis 0 with size 17
    try:
        order_one_hot[selected_order] = 1
    except:
        return order_encodings

    if selected_order in independent_orders:
        actor_one_hot[order_col == selected_order] = 1
        order_type = 0
    elif selected_order in targeted_orders:
        target_col = self_orders[:, 3]
        unique, counts = np.unique(target_col[target_col != 0], return_counts=True)
        selected_target = unique[np.argmax(counts)] 
        target_one_hot[selected_target - 1] = 1
        actor_one_hot[(order_col == selected_order) & (target_col == selected_target)] = 1
        order_type = 1
    else:
        grouping_dist = 96
        xy_cols = self_orders[:, 1:3]
        move_groups = []
        group_sizes = []
        for i in range(xy_cols.shape[0]):
            if order_col[i] == selected_order:
                move_groups.append([i, ])
                group_sizes.append(1)
        largest_group_i = 0
        largest_group_ct = 1
        if len(move_groups) > 1:
            _end = len(move_groups)
            for i in range(_end - 1):
                if len(move_groups[i]) > 0:
                    standard_row = move_groups[i][0]
                    standard_pos = xy_cols[standard_row, :]
                    for j in range(i + 1, _end):
                        if len(move_groups[j]) > 0:
                            compare_row = move_groups[j][0]
                            compare_pos = xy_cols[compare_row, :]
                            distance = np.sqrt(
                                (standard_pos[0] - compare_pos[0])**2 + \
                                (standard_pos[1] - compare_pos[1])**2
                            )
                            if distance < grouping_dist:
                                move_groups[i].append(compare_row)
                                move_groups[j].clear()
                                group_sizes[i] += 1
                                if group_sizes[i] > largest_group_ct:
                                    largest_group_ct = group_sizes[i]
                                    largest_group_i = i;
        largest_move_group = move_groups[largest_group_i]
        sum_x = 0
        sum_y = 0
        for i in largest_move_group:
            actor_one_hot[i] = 1
            sum_x += xy_cols[i, 0]
            sum_y += xy_cols[i, 1]
        avg_scaled_tilepos_x = int(sum_x / (largest_group_ct * 32) * _cache.entity_scale)
        avg_scaled_tilepos_y = int(sum_y / (largest_group_ct * 32) * _cache.entity_scale)
        tilepos_one_hot[avg_scaled_tilepos_y, avg_scaled_tilepos_x] = 1
        order_type = 2
        
    return [order_one_hot, actor_one_hot, target_one_hot, tilepos_one_hot, order_type, (avg_scaled_tilepos_y, avg_scaled_tilepos_x)]


# --------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------ Map ---#
# --------------------------------------------------------------------------------------------------#


"""
Squares the map by adding borders and scales it to be ready for the spatial encoder.
Stacks convolved entity encodings into multiple layers.
"""
def preprocess_map(
    _map, 
    spatial_entity_encodings, 
    entity_positions, 
    scale_to_dim=128, 
    spatial_entity_encoding_sz=8, 
    max_entity_stack_size=4, 
    show_plots=False
):
    map_depth = _map.shape[0]
    map_height = _map.shape[1]
    map_width = _map.shape[2]
    prescale_map_dim = map_height if map_height > map_width else map_width
    entity_tilepositions = entity_positions // 32
    scaled_map_depth = map_depth + spatial_entity_encoding_sz * max_entity_stack_size
    
    # If the dimensions are unequal, make the map square by adding borders before scaling to make
    # normal distance measures true. I don't know if this is important but it seems reasonable.
    # TODO: MIGHT want scaled_map w x h to represent largest possible bw map w x h before adding
    # more maps so that distance is consistent across maps
    if map_width > map_height:
        prescale_map = np.zeros((map_depth, prescale_map_dim, prescale_map_dim))
        prescale_map[:, :map_height, :] = _map
    elif map_height > map_width:
        prescale_map = np.zeros((map_depth, prescale_map_dim, prescale_map_dim))
        prescale_map[:, :, :map_width] = _map
    else:
        prescale_map = _map
    scaled_map = np.zeros((
        scaled_map_depth,
        scale_to_dim,
        scale_to_dim
    ))

    for i in range(map_depth):
        scaled_map[i] = cv2.resize(
            prescale_map[i], (int(scale_to_dim), int(scale_to_dim)), interpolation=cv2.INTER_NEAREST
        )

    # stacking entity encodings into their respective positions on the map
    _cache.entity_scale = scale_to_dim / prescale_map_dim
    entity_tilepositions = (entity_tilepositions * _cache.entity_scale).astype(np.int32)
    stack_sizes = [[0 for _a in range(scale_to_dim)] for _b in range(scale_to_dim)]
    for i in range(spatial_entity_encodings.shape[0]):
        se_enc = spatial_entity_encodings[i]
        tp = entity_tilepositions[i]
        x, y = tp[0], tp[1]
        try:
            stack_size = stack_sizes[y][x]
        except:
            with open('models/supervised_errors.txt', 'a') as f:
                f.write(
                    f'processing::preprocess_map(): bad unit tileposition ({x}, {y})' \
                    f'with scale_to_dim={scale_to_dim}\n'
                )
            continue
        if stack_size < max_entity_stack_size:
            stack_start = map_depth + stack_size * spatial_entity_encoding_sz
            scaled_map[stack_start:stack_start+spatial_entity_encoding_sz, y, x] = se_enc
            stack_sizes[y][x] += 1
        else:
            print(
                "processing::map_preprocessing() entity stack size overflow. Entity encoding lost."
            )

    if show_plots:
        for i in range(map_depth+spatial_entity_encoding_sz):
            plt.imshow(scaled_map[i])
            plt.show()
    
    walkability_mask = scaled_map[0]
    return scaled_map, walkability_mask


# --------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------- General ---#
# --------------------------------------------------------------------------------------------------#


# object shared for caching 
_cache = None
"""
pem = positional encoding memory
"""
class Cache:
    
    def __init__(self):
        self.pem_init()
        self.entity_stats_init()
        self.primes_init()
        self.entity_scale = None
        self.masks_init()

    def masks_init(self):
        attack_types = [37, 38, 39, 41, 43, 44, 47, 62]
        self.attack_mask = np.array([1 if i in attack_types else 0 for i in range(233)]).astype(np.int32)
        burrow_types = [103,]
        self.burrow_mask = np.array([1 if i in burrow_types else 0 for i in range(233)]).astype(np.int32)
        defiler_type = [46, ]
        self.defiler_mask = np.array([1 if i in defiler_type else 0 for i in range(233)]).astype(np.int32)
        other_types = [35,37,38,39,41,42,43,44,45,46,47,62,103]
        self.other_mask = np.array([1 if i in other_types else 0 for i in range(233)]).astype(np.int32)
        self.none_mask = np.zeros(233)

        self.attack_move = np.zeros(17).astype(np.int32)
        self.attack_move[0] = 1
        self.attack_unit = np.zeros(17).astype(np.int32)
        self.attack_unit [1] = 1
        self.burrowing = np.zeros(17).astype(np.int32)
        self.burrowing[2] = 1
        self.consume = np.zeros(17).astype(np.int32)
        self.consume[3] = 1
        self.dark_swarm = np.zeros(17).astype(np.int32)
        self.dark_swarm[4] = 1
        self.plague = np.zeros(17).astype(np.int32)
        self.plague[5] = 1
        self.hold_position = np.zeros(17).astype(np.int32)
        self.hold_position[6] = 1
        self.move = np.zeros(17).astype(np.int32)
        self.move[7] = 1
        self.patrol = np.zeros(17).astype(np.int32)
        self.patrol[8] = 1
        self.unburrowing = np.zeros(17).astype(np.int32)
        self.unburrowing[9] = 1
        self.stop = np.zeros(17).astype(np.int32)
        self.stop[10] = 1
        self.irradiate = np.zeros(17).astype(np.int32)
        self.irradiate[11] = 1
        self.emp = np.zeros(17).astype(np.int32)
        self.emp[12] = 1
        self.defense_matrix = np.zeros(17).astype(np.int32)
        self.defense_matrix[13] = 1
        self.unsieging = np.zeros(17).astype(np.int32)
        self.unsieging[14] = 1
        self.sieging = np.zeros(17).astype(np.int32)
        self.sieging[15] = 1
        self.mine = np.zeros(17).astype(np.int32)
        self.mine[16] = 1
        self.defiler_combined = self.plague & self.dark_swarm & self.consume

    def pem_init(self):
        self.pem_len = 10000
        self.pem_queue = [
            (-i, -i) for i in range(1, self.pem_len+1)
        ]
        self.pem = dict(zip(
            self.pem_queue, 
            [None for i in range(self.pem_len)]
        ))
        self.pem_queue_ptr = 0

    def primes_init(self):
        self.primes = pload(open('data/primes.bin', 'rb'), encoding='bytes')

    def entity_stats_init(self):
        self.max_sqrt_hp = int(np.sqrt(2500))
        self.max_sqrt_en = int(np.sqrt(250))
        self.one_hot_spacing = [233, 1, self.max_sqrt_hp, self.max_sqrt_en, 50, 50, 1, 11]
        self.entity_col_ct = len(self.one_hot_spacing) + 2
        self.one_hot_spacing_cumulative = [
            sum(self.one_hot_spacing[:i]) for i in range(len(self.one_hot_spacing))
        ]
        self.preprocessed_entity_col_ct = sum(self.one_hot_spacing)
        self.entity_permutation_ct = 1
        for spacing in self.one_hot_spacing:
            self.entity_permutation_ct *= spacing + 1

    def pem_cached(self, xy):
        try:
            return self.pem[xy]
        except:
            return None

    def pem_add(self, xy, vec):
        key = self.pem_queue[self.pem_queue_ptr]
        self.pem.pop(key)
        self.pem_queue[self.pem_queue_ptr] = xy
        self.pem[xy] = vec
        self.pem_queue_ptr += 1
        if self.pem_queue_ptr >= self.pem_len:
            self.pem_queue_ptr = 0


"""
The data files can take a while to load. Pickle a single file's worth of data and make it available 
quickly. This is useful for messing with model parameters.
"""
def pickle_file(file_no=0):
    file_data = load_file(file_no)
    if file_data is not None:
        with open('data/pickled_data.bin', 'wb') as f:
            pdump(file_data, f)
        print(f'learn::pickle_file(): file [{file_no}] pickled.')
    else:
        print(f'learn::pickle_file(): bad data file [{file_no}]. not pickling.')


def stats(item):
    if item == 'entity_permutation_ct':
        return _cache.entity_permutation_ct
    elif item == 'entity_col_ct':
        return _cache.preprocessed_entity_col_ct


def init_cache():
    global _cache
    _cache = Cache()


if __name__ == '__main__':
    init_cache()
    test_positional_encoding_array('distance', depth=256, xy_plane_ct=100)
    pass
