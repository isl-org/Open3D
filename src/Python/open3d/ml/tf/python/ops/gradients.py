from .lib import _lib
from tensorflow.python.framework import ops


@ops.RegisterGradient("Open3DVoxelPooling")
def _voxel_pooling_grad(op, grad_pos, grad_feat):
    features_grad = _lib.open3d_voxel_pooling_grad(
        positions=op.inputs[0],
        features=op.inputs[1],
        voxel_size=op.inputs[2],
        pooled_positions=op.outputs[0],
        pooled_features_gradient=grad_feat,
        position_fn=op.get_attr('position_fn'),
        feature_fn=op.get_attr('feature_fn'),
    )
    return [None, features_grad, None]
