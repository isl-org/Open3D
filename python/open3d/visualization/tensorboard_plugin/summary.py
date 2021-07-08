import tensorflow.compat.v2 as tf
from tensorboard.compat.proto import summary_pb2

from open3d.visualization.tensorboard_plugin import metadata
import ipdb


def add_3d(name, data, step=None, max_outputs=3, description=None):
    """
    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A dictionary of `Tensor`s representing 3D data. The following keys
        are supported:
        vertices: shape `[B, N, 3]` where B is the number of point clouds and
            must be same for each key. N is the number of 3D points.
        vertex_colors: shape `[B, N, 3]`
        vertex_normals: shape `[B, N, 3]`
        triangles: shape `[B, Nf, 3]`
        Any of the dimensions may be statically unknown (i.e., `None`).
        Floating point data will be clipped to the range [0,1]. Other data types
        will be clipped into an allowed range for safe casting to uint8, using
        `tf.image.convert_image_dtype`.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
        many images will be emitted at each step. When more than
        `max_outputs` many images are provided, the first `max_outputs` many
        images will be used and the rest silently discarded.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    summary_metadata = metadata.create_summary_metadata(description=description)
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (getattr(tf.summary.experimental, "summary_scope", None) or
                     tf.summary.summary_scope)
    all_ok = True
    with summary_scope(name, "open3d_summary", values=[data, max_outputs,
                                                       step]) as (tag, _):
        tf.debugging.assert_non_negative(max_outputs)
        tf.debugging.Assert(
            'vertices' in data,
            tf.convert_to_tensor("No vertices attribute provided."))
        for att_name, att_val in data.items():
            tf.debugging.Assert(
                att_name in metadata._MESH_PROPERTIES,
                tf.convert_to_tensor("Unsupported attribute " + att_name))
            # TODO: Check attribute shape compatibility. eg: points and colors,
            # etc.
            # TODO: Compress + lazytensor
            tf.debugging.assert_rank(att_val, 3)
            tf.debugging.assert_equal(att_val.shape[-1], 3)
            all_ok = all_ok and tf.summary.write(tag=tag + '_' + att_name,
                                                 tensor=att_val,
                                                 step=step,
                                                 metadata=summary_metadata)
    return all_ok
