from tkinter import W
import numpy as np
from matplotlib import pyplot as plt
from random import randint, sample as randsample, choice
from load import load_entities, load_map


# object shared for caching 
_cache = None


"""
pem = positional encoding memory
"""
class Cache:
    
    def __init__(self):
        self.pem_init()
        self.entity_stats_init()
        
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

    def entity_stats_init(self):
        self.max_sqrt_hp = int(np.sqrt(2500))
        self.max_sqrt_en = int(np.sqrt(250))
        self.one_hot_spacing = [232, 1, self.max_sqrt_hp, self.max_sqrt_en, 50, 50, 1, 10]
        self.entity_col_ct = len(self.one_hot_spacing) + 2
        self.one_hot_spacing_cumulative = [sum(self.one_hot_spacing[:i]) for i in range(len(self.one_hot_spacing))]
        self.preprocessed_entity_col_ct = sum(self.one_hot_spacing)
        self.entity_permutation_ct = 1
        for spacing in self.one_hot_spacing:
            self.entity_permutation_ct *= spacing + 1
            

"""
Adapts the original Transformer positional encoding for fairly large (8192x8192) 2d positional input.
Tested to balance complexity of function and representing nearby x,y points with near positional
vectors, and far-apart x,y points with far positional vectors; the hope is that the function doesn't
have to be precisely parametrically learned by the model since distance is a valuable metric.
"""
def positional_encoding_vector(x, y, depth, w, h):
    # skipping very low minimum distance between vectors in small x, y
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
Creates an array of 2d positional encoding vectors using positional_encoding_vector()
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
def test_positional_encoding_array(test_type='visual', depth=256, width=8192, height=8192, xy_plane_ct=8):
    assert test_type in ['visual', 'distance', 'both']
    assert xy_plane_ct >= 4
    xy_plane_dim = 30
    # always including the 4 corners of the xy space
    y_extremes = (0, height - xy_plane_dim, 0, height - xy_plane_dim)
    x_extremes = (0, width - xy_plane_dim, width - xy_plane_dim, 0)
    # the remaining planes are randomly selected
    yrange_starts = [randint(xy_plane_dim, y_extremes[1] - xy_plane_dim) for _ in range(xy_plane_ct - 4)]
    xrange_starts = [randint(xy_plane_dim, x_extremes[1] - xy_plane_dim) for _ in range(xy_plane_ct - 4)]
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
        print(f'\tmin: {min_near_xy_dist:.9f}, max: {max_near_xy_dist:.9f}, mean: {mean_near_xy_dist:.9f}')
        print('Random (x, y) values:')
        print(f'\tmin: {min_rand_xy_dist:.9f}, max: {max_rand_xy_dist:.9f}, mean: {mean_rand_xy_dist:.9f}')

    
# 0: unit type      (int to one-hot max 233)
# 1: owner          (int to binary one-hot) 
# 2: health         (int -> one hot of sqrt(min(hp, max_bw_hp))) (max bw hp = 2500)
# 3: energy         (int -> one hot of sqrt(min(en, max_bw_en))) (max bw en = 250)
# 4: attack cd      (int -> one hot with maximum 50 (1 learn interval per 2 frames))
# 5: spell cd       (int -> one hot with maximum 50 (1 learn interval per 2 frames))
# 6: is burrowed    (int to binary one-hot)
# 7: armor          (int -> one hot maximum 10)
# 8: position x     (set aside for positional encoding)
# 9: position y     (set aside for positional encoding)   

def preprocessed_entities(entities):
    n = entities.shape[0]
    positions = entities[:, _cache.entity_col_ct-2:]
    entities = entities[:, 0:_cache.entity_col_ct-2]
    entities[:, 2] = (np.sqrt(entities[:, 2])).astype(np.int32)
    entities[:, 3] = (np.sqrt(entities[:, 3])).astype(np.int32)
    entities[:, 4] = entities[:, 4] // 2
    entities[:, 5] = entities[:, 5] // 2
    entities -= 1

    preprocessed = np.zeros((n, _cache.preprocessed_entity_col_ct), dtype=np.int32)
    for i in range(n):
        entity = entities[i, 0:_cache.entity_col_ct-2]
        for j in range(len(_cache.one_hot_spacing)):
            entity_val = entity[j]
            if entity_val < 0:
                continue
            spacing = _cache.one_hot_spacing_cumulative[j]
            preprocessed[i, spacing + entity_val] = 1
    return preprocessed, positional_encoding_array(positions, _cache.preprocessed_entity_col_ct, 8192, 8192)


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
    # test_positional_encoding_array('distance', depth=128, xy_plane_ct=100)
    pass