from .lib import _lib

def tf_batch_subsampling(points, batches_len, sampleDl):
    return _lib.batch_grid_subsampling(
        points, batches_len, sampleDl)

def tf_subsampling(points, sampleDl):
    return _lib.grid_subsampling(
        points, sampleDl)

def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
    return _lib.batch_ordered_neighbors(
        queries, supports, q_batches, s_batches, radius)
