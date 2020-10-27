import numpy as np
from itertools import product


def filter_distance_matrix(m):
    w = (
        check_diag_zero(m)  & 
        check_symmetry(m) & 
        check_triangle_inequality(m)
    )
    return m[w]


def gen_all_matricies_iterative(max_size):
    m = np.array([[0]])
    yield m
    for i in range(1, max_size):
        m = expand(m)
        m = filter_distance_matrix(m)
        yield m

def expand(m):
    old_size = m.shape[1]
    old_n = m.shape[0]
    comb = list(product([0,1], repeat=old_size))
    com_arr = np.array(comb)
    new_m = np.zeros((len(comb), old_n, old_size+1,old_size+1))
    new_m[:,:,:-1,:-1] = m[np.newaxis]
    new_m[:,:,-1,:-1] = com_arr[:, np.newaxis]
    new_m[:,:,:-1,-1] = com_arr[:, np.newaxis]
    new_m = new_m.reshape(-1, old_size+1,old_size+1)
    # print(new_m)
    return new_m


def check_diag_zero(m):
    return (np.diagonal(m, axis1=1, axis2=2) == 0).all(-1)

def check_symmetry(m):
    return (m == np.transpose(m, (0,2,1))).all(-1).all(-1)

def check_triangle_inequality(m):
    return (m <= (m[:,:,:,np.newaxis] + m[:,np.newaxis,:,:]).min(2)).all(-1).all(-1)


def gen_tuple(length, n_colors):
    return np.array(list(product(range(n_colors), repeat=length)))


def gen_dist_m(m):
    return (m[:,np.newaxis] != m[:,:,np.newaxis]).astype(np.float64)

def match_dist_m(m, ref):
    match = (m[:,np.newaxis] == ref[np.newaxis,:]).all(-1).all(-1)
    assert (match.sum(1) == 1).all()
    return match.argmax(1)


# legacy
# def gen_all_matricies(size):
#     comb = product([0,1], repeat=size*size)
#     comb_arr = np.array(list(comb))
#     comb_arr = comb_arr.reshape(-1, size, size)
#     return filter_distance_matrix(comb_arr)


def get_ref_tuples(max_length, n_colors):
    for ref_dist_m in gen_all_matricies_iterative(max_length):
        length = ref_dist_m.shape[-1]
        comb = gen_tuple(length, n_colors)
        comb_dist_m = gen_dist_m(comb)
        sym_type = match_dist_m(comb_dist_m, ref_dist_m)
        yield length, comb, sym_type
