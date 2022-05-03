# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
"""This is a python implementation for the continuous convolutions meant for
debugging and testing the C code.
"""

import numpy as np

# interpolation
LINEAR = 1
NEAREST_NEIGHBOR = 2
LINEAR_BORDER = 3

# coordinate mapping
IDENTITY = 4
BALL_TO_CUBE_RADIAL = 5
BALL_TO_CUBE_VOLUME_PRESERVING = 6

# windows
RECTANGLE = 7
TRAPEZOID = 8
POLY = 9

_convert_parameter_str_dict = {
    'linear': LINEAR,
    'linear_border': LINEAR_BORDER,
    'nearest_neighbor': NEAREST_NEIGHBOR,
    'identity': IDENTITY,
    'ball_to_cube_radial': BALL_TO_CUBE_RADIAL,
    'ball_to_cube_volume_preserving': BALL_TO_CUBE_VOLUME_PRESERVING,
}


def map_cube_to_cylinder(points, inverse=False):
    """maps a cube to a cylinder and vice versa
    The input and output range of the coordinates is [-1,1]. The cylinder axis
    is along z.

    points: numpy array with shape [n,3]
    inverse: If True apply the inverse transform: cylinder -> cube
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    # yapf: disable
    result = np.empty_like(points)

    if inverse:
        for i, p in enumerate(points):
            x, y, z = p
            if np.allclose(p[0:2], np.zeros_like(p[0:2])):
                result[i] = (0,0,z)
            elif np.abs(y) <= x and x > 0:
                result[i] = (np.sqrt(x*x+y*y), 4/np.pi *np.sqrt(x*x+y*y)*np.arctan(y/x), z)
            elif np.abs(y) <= -x and x < 0:
                result[i] = (-np.sqrt(x*x+y*y), -4/np.pi *np.sqrt(x*x+y*y)*np.arctan(y/x), z)
            elif np.abs(x) <= y and y > 0:
                result[i] = (4/np.pi *np.sqrt(x*x+y*y)*np.arctan(x/y), np.sqrt(x*x+y*y), z)
            else: # elif np.abs(x) <= -y and y < 0:
                result[i] = (-4/np.pi *np.sqrt(x*x+y*y)*np.arctan(x/y), -np.sqrt(x*x+y*y), z)
    else:

        for i, p in enumerate(points):
            x, y, z = p
            if np.count_nonzero(p[0:2]) == 0:
                result[i] = (0,0,z)
            elif np.abs(y) <= np.abs(x):
                result[i] = (x*np.cos(y/x*np.pi/4), x*np.sin(y/x*np.pi/4), z)
            else:
                result[i] = (y*np.sin(x/y*np.pi/4), y*np.cos(x/y*np.pi/4), z)

    return result
    # yapf: enable


def map_cylinder_to_sphere(points, inverse=False):
    """maps a cylinder to a sphere and vice versa.
    The input and output range of the coordinates is [-1,1]. The cylinder axis
    is along z.

    points: numpy array with shape [n,3]
    inverse: If True apply the inverse transform: sphere -> cylinder
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    # yapf: disable
    result = np.empty_like(points)

    if inverse:
        for i, p in enumerate(points):
            x, y, z = p
            t = np.linalg.norm(p, ord=2)

            if np.allclose(p, np.zeros_like(p)):
                result[i] = 0,0,0
            elif 5/4*z**2 > (x**2 + y**2):
                s, z = np.sqrt(3*t/(t+np.abs(z))), np.sign(z)*t
                result[i] = s*x, s*y, z

            else: # elif 5/4*z**2 <= (x**2 + y**2):
                s, z = t/np.sqrt(x*x+y*y), 3/2*z
                result[i] = s*x, s*y, z
    else:

        for i, p in enumerate(points):
            x, y, z = p
            if np.allclose(p, np.zeros_like(p)):
                result[i] = 0,0,0
            elif z*z >= x*x + y*y:
                result[i] = ( x*np.sqrt(2/3-(x*x+y*y)/(9*z*z)),
                              y*np.sqrt(2/3-(x*x+y*y)/(9*z*z)),
                              z-(x*x+y*y)/(3*z) )
            else:
                result[i] = ( x*np.sqrt(1-(4*z*z)/(9*(x*x+y*y))),
                              y*np.sqrt(1-(4*z*z)/(9*(x*x+y*y))),
                              2*z/3 )

    return result
    # yapf: enable


def compute_filter_coordinates(pos, filter_xyz_size, inv_extents, offset,
                               align_corners, mapping):
    """Computes the filter coordinates for a single point
    The input to this function are coordinates relative to the point where the
    convolution is evaluated. Coordinates are usually in the range
    [-extent/2,extent/2] with extent as the edge length of the bounding box of
    the filter shape. The output is a coordinate within the filter array, i.e.
    the range is [0, filter_size.xyz], if the point was inside the filter shape.

    The simplest filter shape is a cuboid (mapping=IDENTITY) and the
    transformation is simply [-extent/2,extent/2] -> [0, filter_size.xyz].
    The other type of shape that is implemented is a sphere with
    mapping=BALL_TO_CUBE_RADIAL or mapping=BALL_TO_CUBE_VOLUME_PRESERVING.

    pos: A single 3D point. An array of shape [3] with x,y,z coordinates.

    filter_xyz_size: An array of shape [3], which defines the size of the filter
                     array for the spatial dimensions.

    inv_extents: An array of shape [3], which defines the spatial extent of the
                 filter. The values are the reciprocal of the spatial extent
                 for x,y and z.

    offset: An array of shape [3]. An offset for shifting the center. Can be
            used to implement discrete filters with even filter size.

    align_corners: If True then the voxel centers of the outer voxels
            of the filter array are mapped to the boundary of the filter shape.
            If false then the boundary of the filter array is mapped to the
            boundary of the filter shape.

    mapping: The mapping that is applied to the input coordinates.
             - BALL_TO_CUBE_RADIAL uses radial stretching to map a sphere to
              a cube.
             - BALL_TO_CUBE_VOLUME_PRESERVING is using a more expensive volume
              preserving mapping to map a sphere to a cube.
             - IDENTITY no mapping is applied to the coordinates.

    """
    assert pos.ndim == 1
    assert pos.shape[0] == 3
    assert filter_xyz_size.ndim == 1
    assert all(filter_xyz_size.shape)
    assert inv_extents.ndim == 1
    assert inv_extents.shape[0] == 3
    assert offset.ndim == 1
    assert offset.shape[0] == 3
    p = pos.copy()
    if mapping == BALL_TO_CUBE_RADIAL:
        p *= 2 * inv_extents  # p is now a position in a sphere with radius 1
        abs_max = np.max(np.abs(p))
        if abs_max < 1e-8:
            p = np.zeros_like(p)
        else:
            # map to the unit cube with edge length 1 and range [-0.5,0.5]
            p *= 0.5 * np.sqrt(np.sum(p * p)) / abs_max
    elif mapping == BALL_TO_CUBE_VOLUME_PRESERVING:
        p *= 2 * inv_extents
        p = 0.5 * map_cube_to_cylinder(map_cylinder_to_sphere(p[np.newaxis, :],
                                                              inverse=True),
                                       inverse=True)[0]
    elif mapping == IDENTITY:
        # map to the unit cube with edge length 1 and range [-0.5,0.5]
        p *= inv_extents
    else:
        raise ValueError("Unknown mapping")

    if align_corners:
        p += 0.5
        p *= filter_xyz_size - 1
    else:
        p *= filter_xyz_size
        p += offset

        # integer div
        p += filter_xyz_size // 2

        if filter_xyz_size[0] % 2 == 0:
            p[0] -= 0.5
        if filter_xyz_size[1] % 2 == 0:
            p[1] -= 0.5
        if filter_xyz_size[2] % 2 == 0:
            p[2] -= 0.5

    return p


def window_function(pos, inv_extents, window, window_params):
    r"""Implements 3 types of window functions

    pos: A single 3D point. An array of shape [3] with x,y,z coordinates.

    inv_extents: An array of shape [3], which defines the spatial extent of the
                 filter. The values are the reciprocal of the spatial extent
                 for x,y and z.

    window: The window type. Allowed types are
            -RECTANGLE this just returns 1 everywhere.
            -TRAPEZOID /‾\ plateau with 1 at the center and decays linearly
                       to 0 at the borders.
            -POLY The poly 6 window

    window_params: array with parameters for the windows.
                   Only TRAPEZOID uses this to define the normalized distance
                   from the center at which the linear decay starts.

    """
    assert pos.ndim == 1
    assert pos.shape[0] == 3
    assert inv_extents.ndim == 1
    assert inv_extents.shape[0] == 3
    p = pos.copy()
    if window == RECTANGLE:
        return 1
    elif window == TRAPEZOID:
        p *= 2 * inv_extents  # p is now a position in a sphere with radius 1
        d = np.linalg.norm(p, ord=2)
        d = np.clip(d, 0, 1)

        # the window parameter defines the distance at which the value decreases
        # linearly to 0
        if d > window_params[0]:
            return (1 - d) / (1 - window_params[0])
        else:
            return 1
    elif window == POLY:
        p *= 2 * inv_extents  # p is now a position in a sphere with radius 1
        r_sqr = np.sum(p * p)
        return np.clip((1 - r_sqr)**3, 0, 1)
    else:
        raise ValueError("Unknown window type")


def interpolate(xyz, xyz_size, interpolation):
    """ Computes interpolation weights and indices

    xyz: A single 3D point.

    xyz_size: An array of shape [3], which defines the size of the filter
              array for the spatial dimensions.

    interpolation: One of LINEAR, LINEAR_BORDER, NEAREST_NEIGHBOR.
            LINEAR is trilinear interpolation with coordinate clamping.
            LINEAR_BORDER uses a zero border if outside the range.
            NEAREST_NEIGHBOR uses the nearest neighbor instead of interpolation.

    Returns a tuple with the interpolation weights and the indices
    """
    # yapf: disable
    if interpolation == NEAREST_NEIGHBOR:
        pi = np.round(xyz).astype(np.int32)
        pi = np.clip(pi, np.zeros_like(pi), xyz_size-1)
        idx = pi[2]*xyz_size[0]*xyz_size[1] + pi[1]*xyz_size[0] + pi[0]
        return (1,), ((pi[2],pi[1],pi[0]),)
    elif interpolation == LINEAR_BORDER:
        pi0 = np.floor(xyz).astype(np.int32)
        pi1 = pi0+1
        a = xyz[0]-pi0[0]
        b = xyz[1]-pi0[1]
        c = xyz[2]-pi0[2]

        w = ((1-a)*(1-b)*(1-c),
             (a)*(1-b)*(1-c),
             (1-a)*(b)*(1-c),
             (a)*(b)*(1-c),
             (1-a)*(1-b)*(c),
             (a)*(1-b)*(c),
             (1-a)*(b)*(c),
             (a)*(b)*(c))
        idx=((pi0[2], pi0[1], pi0[0]),
             (pi0[2], pi0[1], pi1[0]),
             (pi0[2], pi1[1], pi0[0]),
             (pi0[2], pi1[1], pi1[0]),
             (pi1[2], pi0[1], pi0[0]),
             (pi1[2], pi0[1], pi1[0]),
             (pi1[2], pi1[1], pi0[0]),
             (pi1[2], pi1[1], pi1[0]))

        w_idx = []
        for w_, idx_ in zip(w,idx):
            if np.any(np.array(idx_) < 0) or idx_[0] >= xyz_size[2] or idx_[1] >= xyz_size[1] or idx_[2] >= xyz_size[0]:
                w_idx.append((0.0, (0,0,0)))
            else:
                w_idx.append((w_,idx_))
        w, idx = zip(*w_idx)
        return w, idx
    elif interpolation == LINEAR:
        pi0 = np.clip(xyz.astype(np.int32), np.zeros_like(xyz, dtype=np.int32), xyz_size-1)
        pi1 = np.clip(pi0+1, np.zeros_like(pi0), xyz_size-1)
        a = xyz[0]-pi0[0]
        b = xyz[1]-pi0[1]
        c = xyz[2]-pi0[2]
        a = np.clip(a, 0, 1)
        b = np.clip(b, 0, 1)
        c = np.clip(c, 0, 1)
        w = ((1-a)*(1-b)*(1-c),
             (a)*(1-b)*(1-c),
             (1-a)*(b)*(1-c),
             (a)*(b)*(1-c),
             (1-a)*(1-b)*(c),
             (a)*(1-b)*(c),
             (1-a)*(b)*(c),
             (a)*(b)*(c))
        idx=((pi0[2], pi0[1], pi0[0]),
             (pi0[2], pi0[1], pi1[0]),
             (pi0[2], pi1[1], pi0[0]),
             (pi0[2], pi1[1], pi1[0]),
             (pi1[2], pi0[1], pi0[0]),
             (pi1[2], pi0[1], pi1[0]),
             (pi1[2], pi1[1], pi0[0]),
             (pi1[2], pi1[1], pi1[0]))
        return w, idx
    else:
        raise ValueError("Unknown interpolation mode")
    # yapf: enable


def cconv(filter, out_positions, extent, offset, inp_positions, inp_features,
          inp_importance, neighbors_index, neighbors_importance,
          neighbors_row_splits, align_corners, coordinate_mapping, normalize,
          interpolation, **kwargs):
    """ Computes the output features of a continuous convolution.

    filter: 5D filter array with shape [depth,height,width,inp_ch, out_ch]

    out_positions: The positions of the output points. The shape is
                   [num_out, 3].

    extents: The spatial extents of the filter in coordinate units.
             This is a 2D array with shape [1,1] or [1,3] or [num_out,1]
             or [num_out,3]

    offset: A single 3D vector used in the filter coordinate
            computation. The shape is [3].

    inp_positions: The positions of the input points. The shape is
                   [num_inp, 3].

    inp_features: The input features with shape [num_inp, in_channels].

    inp_importance: Optional importance for each input point with
                    shape [num_inp]. Set to np.array([]) to disable.

    neighbors_index: The array with lists of neighbors for each
           output point. The start and end of each sublist is defined by
           neighbors_row_splits.

    neighbors_importance: Optional importance for each entry in
           neighbors_index. Set to np.array([]) to disable.

    neighbors_row_splits:   The prefix sum which defines the start
           and end of the sublists in neighbors_index. The size of the
           array is num_out + 1.

    align_corners: If true then the voxel centers of the outer voxels
           of the filter array are mapped to the boundary of the filter shape.
           If false then the boundary of the filter array is mapped to the
           boundary of the filter shape.

    coordinate_mapping: The coordinate mapping function. One of
           IDENTITY, BALL_TO_CUBE_RADIAL, BALL_TO_CUBE_VOLUME_PRESERVING.

    normalize: If true then the result is normalized either by the
           number of points (neighbors_importance is null) or by the sum of
           the respective values in neighbors_importance.

    interpolation: The interpolation mode. Either LINEAR or NEAREST_NEIGHBOR.

    """
    assert filter.ndim == 5
    assert all(filter.shape)
    assert filter.shape[3] == inp_features.shape[-1]
    assert out_positions.ndim == 2
    assert extent.ndim == 2
    assert extent.shape[0] == 1 or extent.shape[0] == out_positions.shape[0]
    assert extent.shape[1] in (1, 3)
    assert offset.ndim == 1 and offset.shape[0] == 3
    assert inp_positions.ndim == 2
    assert inp_positions.shape[0] == inp_features.shape[0]
    assert inp_features.ndim == 2
    assert inp_importance.ndim == 1
    assert (inp_importance.shape[0] == 0 or
            inp_importance.shape[0] == inp_positions.shape[0])
    assert neighbors_importance.ndim == 1
    assert (neighbors_importance.shape[0] == 0 or
            neighbors_importance.shape[0] == neighbors_index.shape[0])
    assert neighbors_index.ndim == 1
    assert neighbors_row_splits.ndim == 1
    assert neighbors_row_splits.shape[0] == out_positions.shape[0] + 1
    coordinate_mapping = _convert_parameter_str_dict[
        coordinate_mapping] if isinstance(coordinate_mapping,
                                          str) else coordinate_mapping
    interpolation = _convert_parameter_str_dict[interpolation] if isinstance(
        interpolation, str) else interpolation

    dtype = inp_features.dtype
    num_out = out_positions.shape[0]
    num_inp = inp_positions.shape[0]
    in_channels = inp_features.shape[-1]
    out_channels = filter.shape[-1]

    inv_extent = 1 / np.broadcast_to(extent, out_positions.shape)

    if inp_importance.shape[0] == 0:
        inp_importance = np.ones([num_inp])

    if neighbors_importance.shape[0] == 0:
        neighbors_importance = np.ones(neighbors_index.shape)

    filter_xyz_size = np.array(list(reversed(filter.shape[0:3])))

    out_features = np.zeros((num_out, out_channels))

    for out_idx, out_pos in enumerate(out_positions):

        neighbors_start = neighbors_row_splits[out_idx]
        neighbors_end = neighbors_row_splits[out_idx + 1]

        outfeat = out_features[out_idx:out_idx + 1]

        n_importance_sum = 0.0

        for inp_idx, n_importance in zip(
                neighbors_index[neighbors_start:neighbors_end],
                neighbors_importance[neighbors_start:neighbors_end]):

            inp_pos = inp_positions[inp_idx]
            relative_pos = inp_pos - out_pos
            coords = compute_filter_coordinates(relative_pos, filter_xyz_size,
                                                inv_extent[out_idx], offset,
                                                align_corners,
                                                coordinate_mapping)

            interp_w, interp_idx = interpolate(coords,
                                               filter_xyz_size,
                                               interpolation=interpolation)

            n_importance_sum += n_importance
            infeat = inp_features[inp_idx:inp_idx +
                                  1] * inp_importance[inp_idx] * n_importance

            filter_value = 0.0
            for w, idx in zip(interp_w, interp_idx):
                filter_value += w * filter[idx]

            outfeat += infeat @ filter_value

        if normalize:
            if n_importance_sum != 0:
                outfeat /= n_importance_sum

    return out_features


def cconv_backprop_filter(filter, out_positions, extent, offset, inp_positions,
                          inp_features, inp_importance, neighbors_index,
                          neighbors_importance, neighbors_row_splits,
                          out_features_gradient, align_corners,
                          coordinate_mapping, normalize, interpolation,
                          **kwargs):
    """This implements the backprop to the filter weights for the cconv.

    out_features_gradient: An array with the gradient for the outputs of the
                           cconv in the forward pass.

    See cconv for more info about the parameters.
    """
    assert filter.ndim == 5
    assert all(filter.shape)
    assert filter.shape[3] == inp_features.shape[-1]
    assert out_positions.ndim == 2
    assert extent.ndim == 2
    assert extent.shape[0] == 1 or extent.shape[0] == out_positions.shape[0]
    assert extent.shape[1] in (1, 3)
    assert offset.ndim == 1 and offset.shape[0] == 3
    assert inp_positions.ndim == 2
    assert inp_positions.shape[0] == inp_features.shape[0]
    assert inp_features.ndim == 2
    assert inp_importance.ndim == 1
    assert (inp_importance.shape[0] == 0 or
            inp_importance.shape[0] == inp_positions.shape[0])
    assert neighbors_importance.ndim == 1
    assert (neighbors_importance.shape[0] == 0 or
            neighbors_importance.shape[0] == neighbors_index.shape[0])
    assert neighbors_index.ndim == 1
    assert neighbors_row_splits.ndim == 1
    assert neighbors_row_splits.shape[0] == out_positions.shape[0] + 1
    coordinate_mapping = _convert_parameter_str_dict[
        coordinate_mapping] if isinstance(coordinate_mapping,
                                          str) else coordinate_mapping
    interpolation = _convert_parameter_str_dict[interpolation] if isinstance(
        interpolation, str) else interpolation

    dtype = inp_features.dtype
    num_out = out_positions.shape[0]
    num_inp = inp_positions.shape[0]
    in_channels = inp_features.shape[-1]
    out_channels = filter.shape[-1]

    inv_extent = 1 / np.broadcast_to(extent, out_positions.shape)

    if inp_importance.shape[0] == 0:
        inp_importance = np.ones([num_inp])

    if neighbors_importance.shape[0] == 0:
        neighbors_importance = np.ones(neighbors_index.shape)

    filter_xyz_size = np.array(list(reversed(filter.shape[0:3])))

    filter_backprop = np.zeros_like(filter)

    for out_idx, out_pos in enumerate(out_positions):

        neighbors_start = neighbors_row_splits[out_idx]
        neighbors_end = neighbors_row_splits[out_idx + 1]

        n_importance_sum = 1.0
        if normalize:
            n_importance_sum = 0.0
            for inp_idx, n_importance in zip(
                    neighbors_index[neighbors_start:neighbors_end],
                    neighbors_importance[neighbors_start:neighbors_end]):

                inp_pos = inp_positions[inp_idx]
                relative_pos = inp_pos - out_pos
                n_importance_sum += n_importance

        normalizer = 1 / n_importance_sum if n_importance_sum != 0.0 else 1

        outfeat_grad = normalizer * out_features_gradient[out_idx:out_idx + 1]

        for inp_idx, n_importance in zip(
                neighbors_index[neighbors_start:neighbors_end],
                neighbors_importance[neighbors_start:neighbors_end]):

            inp_pos = inp_positions[inp_idx]
            relative_pos = inp_pos - out_pos
            coords = compute_filter_coordinates(relative_pos, filter_xyz_size,
                                                inv_extent[out_idx], offset,
                                                align_corners,
                                                coordinate_mapping)

            interp_w, interp_idx = interpolate(coords,
                                               filter_xyz_size,
                                               interpolation=interpolation)

            infeat = inp_features[inp_idx:inp_idx +
                                  1] * inp_importance[inp_idx] * n_importance

            for w, idx in zip(interp_w, interp_idx):
                filter_backprop[idx] += w * (infeat.T @ outfeat_grad)

    return filter_backprop


def cconv_transpose(filter, out_positions, out_importance, extent, offset,
                    inp_positions, inp_features, inp_neighbors_index,
                    inp_neighbors_importance, inp_neighbors_row_splits,
                    neighbors_index, neighbors_importance, neighbors_row_splits,
                    align_corners, coordinate_mapping, normalize, interpolation,
                    **kwargs):
    """Computes the output features of a transpose continuous convolution.
    This is also used for computing the backprop to the input features for the
    normal cconv.

    filter: 5D filter array with shape [depth,height,width,inp_ch, out_ch]

    out_positions: The positions of the output points. The shape is
                   [num_out, 3].

    inp_importance: Optional importance for each output point with
                    shape [num_out]. Set to np.array([]) to disable.

    extents: The spatial extents of the filter in coordinate units.
             This is a 2D array with shape [1,1] or [1,3] or [num_inp,1]
             or [num_inp,3]

    offset: A single 3D vector used in the filter coordinate
            computation. The shape is [3].

    inp_positions: The positions of the input points. The shape is
                   [num_inp, 3].

    inp_features: The input features with shape [num_inp, in_channels].

    inp_neighbors_index: The array with lists of neighbors for each
           input point. The start and end of each sublist is defined by
           inp_neighbors_row_splits.

    inp_neighbors_importance: Optional importance for each entry in
           inp_neighbors_index. Set to np.array([]) to disable.

    inp_neighbors_row_splits:   The prefix sum which defines the start
           and end of the sublists in inp_neighbors_index. The size of the
           array is num_inp + 1.

    neighbors_index: The array with lists of neighbors for each
           output point. The start and end of each sublist is defined by
           neighbors_row_splits.

    neighbors_importance: Optional importance for each entry in
           neighbors_index. Set to np.array([]) to disable.

    neighbors_row_splits:   The prefix sum which defines the start
           and end of the sublists in neighbors_index. The size of the
           array is num_out + 1.

    align_corners: If true then the voxel centers of the outer voxels
           of the filter array are mapped to the boundary of the filter shape.
           If false then the boundary of the filter array is mapped to the
           boundary of the filter shape.

    coordinate_mapping: The coordinate mapping function. One of
           IDENTITY, BALL_TO_CUBE_RADIAL, BALL_TO_CUBE_VOLUME_PRESERVING.

    normalize: If true then the result is normalized either by the
           number of points (neighbors_importance is null) or by the sum of
           the respective values in neighbors_importance.

    interpolation: The interpolation mode. Either LINEAR or NEAREST_NEIGHBOR.

    """
    assert filter.ndim == 5
    assert all(filter.shape)
    assert filter.shape[3] == inp_features.shape[-1]
    assert out_positions.ndim == 2
    assert out_importance.ndim == 1
    assert (out_importance.shape[0] == 0 or
            out_importance.shape[0] == out_positions.shape[0])
    assert extent.ndim == 2
    assert extent.shape[0] == 1 or extent.shape[0] == inp_positions.shape[0]
    assert extent.shape[1] in (1, 3)
    assert offset.ndim == 1 and offset.shape[0] == 3
    assert inp_positions.ndim == 2
    assert inp_positions.shape[0] == inp_features.shape[0]
    assert inp_features.ndim == 2
    assert inp_neighbors_index.ndim == 1
    assert inp_neighbors_importance.ndim == 1
    assert (inp_neighbors_importance.shape[0] == 0 or
            inp_neighbors_importance.shape[0] == inp_neighbors_index.shape[0])
    assert inp_neighbors_row_splits.ndim == 1
    assert inp_neighbors_row_splits.shape[0] == inp_positions.shape[0] + 1
    assert neighbors_index.ndim == 1
    assert neighbors_importance.ndim == 1
    assert (neighbors_importance.shape[0] == 0 or
            neighbors_importance.shape[0] == neighbors_index.shape[0])
    assert neighbors_row_splits.ndim == 1
    assert neighbors_row_splits.shape[0] == out_positions.shape[0] + 1
    assert neighbors_index.shape[0] == inp_neighbors_index.shape[0]
    coordinate_mapping = _convert_parameter_str_dict[
        coordinate_mapping] if isinstance(coordinate_mapping,
                                          str) else coordinate_mapping
    interpolation = _convert_parameter_str_dict[interpolation] if isinstance(
        interpolation, str) else interpolation

    dtype = inp_features.dtype
    num_out = out_positions.shape[0]
    num_inp = inp_positions.shape[0]
    in_channels = inp_features.shape[-1]
    out_channels = filter.shape[
        -1]  # filter shape is [depth,height,width, in_ch, out_ch]

    inv_extent = 1 / np.broadcast_to(extent, inp_positions.shape)

    if out_importance.shape[0] == 0:
        out_importance = np.ones([num_out])

    if neighbors_importance.shape[0] == 0:
        neighbors_importance = np.ones(neighbors_index.shape)

    if inp_neighbors_importance.shape[0] == 0:
        inp_neighbors_importance = np.ones(inp_neighbors_index.shape)

    if normalize:
        inp_n_importance_sums = np.zeros_like(inp_neighbors_row_splits[:-1],
                                              dtype=out_positions.dtype)
        for inp_idx, inp_pos in enumerate(inp_positions):

            inp_neighbors_start = inp_neighbors_row_splits[inp_idx]
            inp_neighbors_end = inp_neighbors_row_splits[inp_idx + 1]

            for out_idx, n_importance in zip(
                    inp_neighbors_index[inp_neighbors_start:inp_neighbors_end],
                    inp_neighbors_importance[
                        inp_neighbors_start:inp_neighbors_end]):
                inp_n_importance_sums[inp_idx] += n_importance

    filter_xyz_size = np.array(list(reversed(filter.shape[0:3])))

    out_features = np.zeros((num_out, out_channels))

    for out_idx, out_pos in enumerate(out_positions):

        neighbors_start = neighbors_row_splits[out_idx]
        neighbors_end = neighbors_row_splits[out_idx + 1]

        for inp_idx, n_importance in zip(
                neighbors_index[neighbors_start:neighbors_end],
                neighbors_importance[neighbors_start:neighbors_end]):

            inp_pos = inp_positions[inp_idx]

            normalizer = 1
            if normalize:
                n_importance_sum = inp_n_importance_sums[inp_idx]
                if n_importance_sum != 0.0:
                    normalizer = 1 / n_importance_sum

            relative_pos = out_pos - inp_pos
            coords = compute_filter_coordinates(relative_pos, filter_xyz_size,
                                                inv_extent[inp_idx], offset,
                                                align_corners,
                                                coordinate_mapping)

            infeat = normalizer * inp_features[inp_idx:inp_idx +
                                               1] * n_importance

            interp_w, interp_idx = interpolate(coords,
                                               filter_xyz_size,
                                               interpolation=interpolation)

            filter_value = 0.0
            for w, idx in zip(interp_w, interp_idx):
                filter_value += w * filter[idx]

            out_features[out_idx:out_idx + 1] += infeat @ filter_value

    out_features *= out_importance[:, np.newaxis]

    return out_features


def cconv_transpose_backprop_filter(
        filter, out_positions, out_importance, extent, offset, inp_positions,
        inp_features, inp_neighbors_index, inp_neighbors_importance,
        inp_neighbors_row_splits, neighbors_index, neighbors_importance,
        neighbors_row_splits, out_features_gradient, align_corners,
        coordinate_mapping, normalize, interpolation, **kwargs):
    """This implements the backprop to the filter weights for the transpose
    cconv.

    out_features_gradient: An array with the gradient for the outputs of the
                           cconv in the forward pass.

    See cconv_transpose for more info about the parameters.
    """
    assert filter.ndim == 5
    assert all(filter.shape)
    assert filter.shape[3] == inp_features.shape[-1]
    assert out_positions.ndim == 2
    assert extent.ndim == 2
    assert extent.shape[0] == 1 or extent.shape[0] == inp_positions.shape[0]
    assert extent.shape[1] in (1, 3)
    assert offset.ndim == 1 and offset.shape[0] == 3
    assert inp_positions.ndim == 2
    assert inp_positions.shape[0] == inp_features.shape[0]
    assert inp_features.ndim == 2
    assert out_importance.ndim == 1
    assert (out_importance.shape[0] == 0 or
            out_importance.shape[0] == out_positions.shape[0])
    assert inp_neighbors_index.ndim == 1
    assert inp_neighbors_importance.ndim == 1
    assert (inp_neighbors_importance.shape[0] == 0 or
            inp_neighbors_importance.shape[0] == inp_neighbors_index.shape[0])
    assert inp_neighbors_row_splits.ndim == 1
    assert inp_neighbors_row_splits.shape[0] == inp_positions.shape[0] + 1
    assert neighbors_index.ndim == 1
    assert neighbors_importance.ndim == 1
    assert (neighbors_importance.shape[0] == 0 or
            neighbors_importance.shape[0] == neighbors_index.shape[0])
    assert neighbors_row_splits.ndim == 1
    assert neighbors_row_splits.shape[0] == out_positions.shape[0] + 1
    assert neighbors_index.shape[0] == inp_neighbors_index.shape[0]
    coordinate_mapping = _convert_parameter_str_dict[
        coordinate_mapping] if isinstance(coordinate_mapping,
                                          str) else coordinate_mapping
    interpolation = _convert_parameter_str_dict[interpolation] if isinstance(
        interpolation, str) else interpolation

    dtype = inp_features.dtype
    num_out = out_positions.shape[0]
    num_inp = inp_positions.shape[0]
    in_channels = inp_features.shape[-1]
    out_channels = filter.shape[-1]

    inv_extent = 1 / np.broadcast_to(extent, inp_positions.shape)

    if out_importance.shape[0] == 0:
        out_importance = np.ones([num_out])

    if neighbors_importance.shape[0] == 0:
        neighbors_importance = np.ones(neighbors_index.shape)

    if inp_neighbors_importance.shape[0] == 0:
        inp_neighbors_importance = np.ones(inp_neighbors_index.shape)

    if normalize:
        inp_n_importance_sums = np.zeros_like(inp_neighbors_row_splits[:-1],
                                              dtype=out_positions.dtype)
        for inp_idx, inp_pos in enumerate(inp_positions):

            inp_neighbors_start = inp_neighbors_row_splits[inp_idx]
            inp_neighbors_end = inp_neighbors_row_splits[inp_idx + 1]

            for out_idx, n_importance in zip(
                    inp_neighbors_index[inp_neighbors_start:inp_neighbors_end],
                    inp_neighbors_importance[
                        inp_neighbors_start:inp_neighbors_end]):
                inp_n_importance_sums[inp_idx] += n_importance

    filter_xyz_size = np.array(list(reversed(filter.shape[0:3])))

    filter_backprop = np.zeros_like(filter)

    for out_idx, out_pos in enumerate(out_positions):

        neighbors_start = neighbors_row_splits[out_idx]
        neighbors_end = neighbors_row_splits[out_idx + 1]

        outfeat_grad = out_features_gradient[out_idx:out_idx +
                                             1] * out_importance[out_idx]

        for inp_idx, n_importance in zip(
                neighbors_index[neighbors_start:neighbors_end],
                neighbors_importance[neighbors_start:neighbors_end]):

            inp_pos = inp_positions[inp_idx]

            normalizer = 1
            if normalize:
                n_importance_sum = inp_n_importance_sums[inp_idx]
                if n_importance_sum != 0.0:
                    normalizer = 1 / n_importance_sum

            relative_pos = out_pos - inp_pos
            coords = compute_filter_coordinates(relative_pos, filter_xyz_size,
                                                inv_extent[inp_idx], offset,
                                                align_corners,
                                                coordinate_mapping)

            interp_w, interp_idx = interpolate(coords,
                                               filter_xyz_size,
                                               interpolation=interpolation)

            infeat = normalizer * inp_features[inp_idx:inp_idx +
                                               1] * n_importance

            for w, idx in zip(interp_w, interp_idx):
                filter_backprop[idx] += w * (outfeat_grad.T @ infeat).T

    return filter_backprop
