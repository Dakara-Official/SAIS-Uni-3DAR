import numpy as np
import numba
from numba import njit
from scipy.spatial.transform import Rotation as R

subcell_orders = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)


def kabsch_algorithm(P, Q):
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def assert_points_rotation_invariant(original_points, transformed_points):
    R = kabsch_algorithm(original_points, transformed_points)

    rotated_points = original_points @ R.T

    assert np.allclose(
        rotated_points, transformed_points, atol=1e-7
    ), "Transformed points are not a pure rotation of the original points"


@njit
def interleave_bits(val):
    assert val < 1024
    val = (val | (val << 16)) & 0x030000FF
    val = (val | (val << 8)) & 0x0300F00F
    val = (val | (val << 4)) & 0x030C30C3
    val = (val | (val << 2)) & 0x09249249
    return val


@njit
def z_order_3d_batch(coord):
    coord = coord.astype(np.int32)
    n = coord.shape[0]
    r = np.zeros(n, dtype=np.int32)
    for i in range(n):
        a = interleave_bits(coord[i, 0])
        b = interleave_bits(coord[i, 1])
        c = interleave_bits(coord[i, 2])
        r[i] = a | (b << 1) | (c << 2)
    return r


@njit
def sample_from_duplicate(grid_pos):
    zorder = z_order_3d_batch(grid_pos)
    mapper = {}
    for i in range(zorder.shape[0]):
        key = zorder[i]
        if key in mapper:
            mapper[key].append(i)
        else:
            mapper[key] = numba.typed.List((i,))
    atom_indices = []
    for key in mapper:
        L = mapper[key]
        random_index = np.random.randint(0, len(L))
        cur_idx = L[random_index]
        atom_indices.append(cur_idx)
    sorted_indices = np.sort(atom_indices)
    return sorted_indices


@njit
def reorder_atom_pos(atom_pos, tree_last_level_pos):
    atom_zorder = z_order_3d_batch(atom_pos)
    tree_zorder = z_order_3d_batch(tree_last_level_pos)
    atom_indices_mapper = {}
    for i in range(atom_zorder.shape[0]):
        atom_indices_mapper[atom_zorder[i]] = i
    mapped_indices = []
    for i in range(tree_zorder.shape[0]):
        mapped_indices.append(atom_indices_mapper[tree_zorder[i]])
    return np.array(mapped_indices)


def normalize_pos(atom_pos, random_rotation_prob):
    atom_pos = atom_pos.astype(np.float64)
    atom_pos = atom_pos - atom_pos.mean(axis=0)
    use_rot = np.random.rand() < random_rotation_prob
    if use_rot:
        random_quat = np.random.randn(4)
        random_quat /= np.linalg.norm(random_quat)
        rot = R.from_quat(random_quat).as_matrix()
        pos_rotated = atom_pos @ rot.T
        assert_points_rotation_invariant(atom_pos, pos_rotated)
        return pos_rotated, rot.T
    else:
        return atom_pos, np.eye(3)


@njit
def get_subcell_types(cur_cells, atom_zorders, cur_level):
    shift = subcell_orders.reshape(1, 8, 3)
    divide_factor = 8 ** (cur_level - 1)
    normalize_atom_zorder = (atom_zorders // divide_factor) * divide_factor
    atom_zorders_cnt = dict()
    for i in range(normalize_atom_zorder.shape[0]):
        if normalize_atom_zorder[i] in atom_zorders_cnt:
            atom_zorders_cnt[normalize_atom_zorder[i]] += 1
        else:
            atom_zorders_cnt[normalize_atom_zorder[i]] = 1
    sub_cell_len = 2 ** (cur_level - 1)
    expand_grids = cur_cells.reshape(-1, 1, 3) + shift * sub_cell_len
    expand_grids = expand_grids.reshape(-1, 3)
    expand_zorder = z_order_3d_batch(expand_grids)
    cell_types = np.zeros(cur_cells.shape[0] * 8, dtype=np.int32)
    for i in range(expand_zorder.shape[0]):
        if expand_zorder[i] in atom_zorders_cnt:
            cell_types[i] = atom_zorders_cnt[expand_zorder[i]]
    type = (cell_types.reshape(-1, 8) > 0).astype(np.int32)
    sub_grids = expand_grids[cell_types > 0]
    sub_grids_cnt = cell_types[cell_types > 0]
    return type, sub_grids, sub_grids_cnt


def expand_tree_topdown(
    max_expand_level,
    atom_grid_pos,
):
    atom_grid_zorder = z_order_3d_batch(atom_grid_pos)
    # count only once for duplicated zorder
    atom_grid_zorder = np.unique(atom_grid_zorder)
    # append root nodes
    root_num_cells = 1
    root_grids = np.zeros((root_num_cells, 3), dtype=np.int32)
    root_types, sub_grids, sub_grids_cnt = get_subcell_types(
        root_grids, atom_grid_zorder, max_expand_level
    )
    bit = 1 << np.arange(8).reshape(8, 1)
    root_types = root_types @ bit

    total_grids = [root_grids]
    total_types = [root_types.reshape(-1)]
    total_counts = [np.full(1, atom_grid_zorder.shape[0], dtype=np.int32)]

    for i in range(max_expand_level - 1, 0, -1):
        cur_grids = sub_grids
        cur_grids_cnt = sub_grids_cnt
        cur_types, sub_grids, sub_grids_cnt = get_subcell_types(
            cur_grids, atom_grid_zorder, i
        )
        cur_types = cur_types @ bit
        total_grids.append(cur_grids)
        total_types.append(cur_types.reshape(-1))
        total_counts.append(cur_grids_cnt)

    total_grids.append(sub_grids)

    cur_types = np.full(sub_grids.shape[0], -1, dtype=np.int32)
    total_types.append(cur_types)
    total_counts.append(np.ones(sub_grids.shape[0], dtype=np.int32))

    return total_grids, total_types, total_counts
