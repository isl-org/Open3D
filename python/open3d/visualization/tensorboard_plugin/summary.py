import threading
import os
import socket
import time
import queue

import numpy as np

import tensorflow.compat.v2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.backend.event_processing.plugin_asset_util import PluginDirectory
from tensorboard.util import lazy_tensor_creator
from tensorflow.experimental import dlpack as tf_dlpack

import torch
from torch.utils import dlpack as torch_dlpack

import open3d as o3d
_log = o3d.log
from open3d.visualization.tensorboard_plugin import metadata
from open3d.visualization.tensorboard_plugin import plugin_data_pb2
import ipdb


class _AsyncDataWriter:
    """ Write binary data to file asynchronously. Data buffers and files are
    queued with ``enqueue()`` and actual writing is done in a separate
    thread. GFile (``tf.io.gfile``) is used for writing to local and remote
    (Google cloud storage with gs:// URI and HDFS with hdfs:// URIs) locations.
    The filename format is ``{tagfilepath}.{current time
    (s)}.{hostname}.{Process ID}{filename_extension}`` following the
    TensorFlow event file name format.

    This class is thread safe. A single global object is created when this module is
    imported by each process.
    """

    def __init__(self,
                 max_queue=10,
                 flush_secs=120,
                 filename_extension='.msgpack'):
        """
        Args:
            max_queue (int): enqueue will block if more than ``max_queue``
                writes are pending.
            flush_secs (Number): Data is flushed to disk / network periodically
                with this interval. Note that the data may still be in an OS buffer
                and not on disk.
            filename_extension (str): Extension for binary file.
        """
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._next_flush_time = time.time() + self._flush_secs
        self._filename_extension = filename_extension
        self._write_queue = queue.Queue(maxsize=self._max_queue)
        self._file_handles = dict()
        self._file_next_write_pos = dict()
        # protects _file_handles and _file_next_write_pos
        self._file_lock = threading.Lock()
        self._writer_thread = threading.Thread(target=self._writer,
                                               name="Open3DDataWriter",
                                               daemon=True)
        self._writer_thread.start()

    def _writer(self):
        """Writer thread main function. Since this is a daemon thread, it will
        not block program exit and file handles should be safely closed."""

        while True:
            tagfilepath, data = self._write_queue.get()
            _log.debug(
                f"Writing {len(data)}b data at "
                f"{tagfilepath}+{self._file_handles[tagfilepath].tell()}")
            with self._file_lock:
                handle = self._file_handles[tagfilepath]
            handle.write(data)  # release lock for expensive I/O op
            self._write_queue.task_done()
            if time.time() > self._next_flush_time:
                file_handle_iter = iter(self._file_handles.values())
                try:
                    while True:
                        with self._file_lock:
                            handle = next(file_handle_iter)
                        handle.flush()  # release lock for expensive I/O op
                except (StopIteration, RuntimeError):
                    # RuntimeError: possible race condition in dict iterator,
                    # but PEP3106 guarantees no coruption. Try again later.
                    pass
                self._next_flush_time += self._flush_secs

    def enqueue(self, tagfilepath, data):
        """Add a write job to the write queue.

        Args:
            tagfilepath (str): Full file pathname for data. A suffix will be
                added to get the complete filename. An empty value indicates
                writing is over and the writer thread should join.
            data (bytes): Data buffer

        Returns:
            Tuple of filename and location (in bytes) where the data will be
            written.
        """
        with self._file_lock:
            if tagfilepath not in self._file_handles:
                # summary.writer.EventFileWriter name format
                fullfilepath = "{}.{}.{}.{}{}".format(tagfilepath,
                                                      int(time.time()),
                                                      socket.gethostname(),
                                                      os.getpid(),
                                                      self._filename_extension)
                tf.io.gfile.makedirs(os.path.dirname(fullfilepath))
                self._file_handles[tagfilepath] = tf.io.gfile.GFile(
                    fullfilepath, 'wb')
                _log.debug(f"msgpack file {fullfilepath} opened for writing.")
                this_write_loc = 0
                self._file_next_write_pos[tagfilepath] = len(data)
            else:
                this_write_loc = self._file_next_write_pos[tagfilepath]
                self._file_next_write_pos[tagfilepath] += len(data)
                fullfilepath = self._file_handles[tagfilepath].name
        # Blocks till queue has available slot.
        self._write_queue.put((tagfilepath, data), block=True)
        return os.path.basename(fullfilepath), this_write_loc


# Single global writer per process
_async_data_writer = _AsyncDataWriter()


def _to_o3d(tensor):
    """Convert Tensorflow, PyTorch and Numpy tensors to Open3D tensor without
    copying.
    """

    if isinstance(tensor, o3d.core.Tensor):
        return tensor
    if isinstance(tensor, tf.Tensor):
        return o3d.core.Tensor.from_dlpack(tf_dlpack.to_dlpack(tensor))
    if isinstance(tensor, torch.Tensor):
        return o3d.core.Tensor.from_dlpack(torch_dlpack.to_dlpack(tensor))
    return o3d.core.Tensor.from_numpy(np.asarray(tensor))


def _to_uint8(color_data):
    """
    Args:
        color_data: o3d.core.Tensor [B,N,3] with any dtype. Float dtypes are
        expected to have values in [0,1] and 8 bit Int dtypes in [0,255] and 16
        bit Int types in [0,2^16-1]
    Returns:
        color_data with the same shape, but as uint8 dtype.

    """
    if color_data.dtype == o3d.core.uint8:
        return color_data
    elif color_data.dtype == o3d.core.uint16:
        return (color_data / 256).to(dtype=o3d.core.uint8)
    else:
        return (255 * color_data).to(dtype=o3d.core.uint8)


def _to_integer(tensor):
    """Test converting to scalar integer (np.int64) and return it. Return None
    on failure.
    """
    try:
        val = np.int64(tensor)
        if val.ndim != 0:
            return None
        return val
    except (ValueError, RuntimeError):
        return None


def _preprocess(prop, tensor, step, max_outputs, geometry_metadata):
    """Data conversion and other preprocessing.
    TODO(ssheorey): Convert to half precision, compression, etc.
    """
    # Check if property is reference to prior step
    step_ref = _to_integer(tensor)
    if step_ref is not None:
        if step_ref < 0 or step_ref >= step:
            raise ValueError(f"Out of order step refernce {step_ref} for "
                             f"property {prop} at step {step}")
        geometry_metadata.property_references.add(
            geometry_property=plugin_data_pb2.Open3DPluginData.GeometryProperty.
            Value(prop),
            step_ref=step_ref)
        return None
    if tensor.ndim == 2:  # batch_size = 1
        save_tensor = _to_o3d(tensor)
        save_tensor.reshape((1,) + tuple(save_tensor.shape))
    elif tensor.ndim == 3:
        save_tensor = _to_o3d(tensor[:max_outputs, ...])
    else:
        raise ValueError(f"Property {prop} tensor should be of shape "
                         f"BxNxNp but is {tensor.shape}.")

    # Datatype conversion
    if prop.endswith("_colors"):
        save_tensor = _to_uint8(save_tensor)  # includes scaling
    elif prop.endswith("_indices"):
        save_tensor = save_tensor.to(dtype=o3d.core.int32)
    else:
        save_tensor = save_tensor.to(dtype=o3d.core.float32)

    return save_tensor


def _write_geometry_data(write_dir, tag, step, data, max_outputs=3):
    """Serialize and write geometry data for a tag. Data is written to a
    separate file per tag.
    TODO: Add version specific reader / writer

    Args:
        write_dir (str): Path of folder to write data file.
        tag (str): Full name for geometry.
        step (int): Iteration / step count.
        data (dict): Property name to tensor mapping.
        max_outputs (int): Only the first `max_samples` data points in each
            batch will be saved.

    Returns:
        A comma separated data location string with the format
        f"{filename},{write_location},{write_size}"
    """

    if not isinstance(data, dict):
        raise TypeError(
            "data should be a dict of geometry property names and tensors.")
    unknown_props = [
        prop for prop in data if prop not in metadata.GEOMETRY_PROPERTY_DIMS
    ]
    if unknown_props:
        raise ValueError("Unknown geometry properties in data: " +
                         str(unknown_props))
    if "vertex_positions" not in data:
        raise ValueError("Primary key 'vertex_positions' not provided.")
    if max_outputs < 1:
        raise ValueError(
            f"max_outputs ({max_outputs}) should be a non-negative integer.")
    max_outputs = int(max_outputs)

    batch_size = None
    n_vertices = None
    n_triangles = None
    n_lines = None
    vertex_data = {}
    triangle_data = {}
    line_data = {}
    geometry_metadata = plugin_data_pb2.Open3DPluginData(
        version=metadata._VERSION)
    for prop, tensor in data.items():
        if prop in ('vertex_positions',) + metadata.VERTEX_PROPERTIES:
            prop_name = prop[7:]
            vertex_data[prop_name] = _preprocess(prop, tensor, step,
                                                 max_outputs, geometry_metadata)
            if vertex_data[prop_name] is None:  # Step reference
                del vertex_data[prop_name]
                continue
            if batch_size is None:  # Get tensor dims from earlier property
                batch_size, n_vertices, _ = vertex_data[prop_name].shape
            exp_shape = (batch_size, n_vertices,
                         metadata.GEOMETRY_PROPERTY_DIMS[prop])
            if tuple(vertex_data[prop_name].shape) != exp_shape:
                raise ValueError(
                    f"Property {prop} tensor should be of shape "
                    f"{exp_shape} but is {vertex_data[prop_name].shape}.")

        elif prop in ('triangle_indices',) + metadata.TRIANGLE_PROPERTIES:
            prop_name = prop[9:]
            triangle_data[prop_name] = _preprocess(prop, tensor, step,
                                                   max_outputs,
                                                   geometry_metadata)
            if triangle_data[prop_name] is None:  # Step reference
                del triangle_data[prop_name]
                continue
            if n_triangles is None:  # Get tensor dims from earlier property
                _, n_triangles, _ = triangle_data[prop_name].shape
            exp_shape = (batch_size, n_triangles,
                         metadata.GEOMETRY_PROPERTY_DIMS[prop])
            if tuple(triangle_data[prop_name].shape) != exp_shape:
                raise ValueError(
                    f"Property {prop} tensor should be of shape "
                    f"{exp_shape} but is {triangle_data[prop_name].shape}.")

        elif prop in ('line_indices',) + metadata.LINE_PROPERTIES:
            line_data[prop_name] = _preprocess(prop, tensor, step, max_outputs,
                                               geometry_metadata)
            if line_data[prop_name] is None:  # Step reference
                del vertex_data[prop_name]
                continue
            if n_lines is None:  # Get tensor dims from earlier property
                _, n_lines, _ = line_data[prop_name].shape
            exp_shape = (batch_size, n_lines,
                         metadata.GEOMETRY_PROPERTY_DIMS[prop_name])
            if tuple(line_data[prop_name].shape) != exp_shape:
                raise ValueError(
                    f"Property {prop} tensor should be of shape "
                    f"{exp_shape} but is {line_data[prop_name].shape}.")

    vertices = vertex_data.pop("positions",
                               o3d.core.Tensor((), dtype=o3d.core.int32))
    faces = triangle_data.pop("indices",
                              o3d.core.Tensor((), dtype=o3d.core.int32))
    lines = line_data.pop("indices", o3d.core.Tensor((), dtype=o3d.core.int32))
    for b in range(batch_size):
        bc = o3d.io.rpc.BufferConnection()
        o3d.io.rpc.set_mesh_data(
            vertices=vertices[b, :, :] if vertices.ndim == 3 else vertices,
            path=tag,
            time=step,
            layer="",
            vertex_attributes={
                prop: tensor[b, :, :] for prop, tensor in vertex_data.items()
            },
            faces=faces[b, :, :] if faces.ndim == 3 else faces,
            face_attributes={
                prop: tensor[b, :, :] for prop, tensor in triangle_data.items()
            },
            lines=lines[b, :, :] if lines.ndim == 3 else lines,
            line_attributes={
                prop: tensor[b, :, :] for prop, tensor in line_data.items()
            },
            connection=bc)
        # TODO: This returns a copy instead of the original. Benchmark
        data_buffer = bc.get_buffer()
        filename, this_write_location = _async_data_writer.enqueue(
            os.path.join(write_dir, tag.replace('/', '-')), data_buffer)
        if b == 0:
            geometry_metadata.batch_index.filename = filename
        geometry_metadata.batch_index.start_size.add(start=this_write_location,
                                                     size=len(data_buffer))
    return geometry_metadata.SerializeToString()


def add_3d(name, data, step=None, max_outputs=1, description=None):
    """Write 3D geometry data as summary.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A dictionary of ``Tensor``s representing 3D data. Tensorflow,
        PyTorch, Numpy and Open3D tensors are supported. The following keys
        are supported:

          - ``vertex_positions``: shape `(B, N, 3)` where B is the number of
                point clouds and must be same for each key. N is the number of
                3D points. WIll be cast to ``float32``.
          - ``vertex_colors``: shape `(B, N, 3)` WIll be converted to ``uint8``.
          - ``vertex_normals``: shape `(B, N, 3)` WIll be cast to ``float32``.
          - ``vertex_uvs``: shape `(B, N, 2)`
          - ``triangle_indices``: shape `(B, Nf, 3)`. Will be cast to ``uint32``.
          - ``line_indices``: shape `(B, Nl, 2)`. Will be cast to ``uint32``.

        For batch_size B=1, the tensors may be rank 2 instead of rank 3.
        Floating point color data will be clipped to the range [0,1] and
        converted to uint8 range [0,255]. Other data types will be clipped into
        an allowed range for safe casting to uint8.

        Any data tensor may be replaced by an int scalar referring to a
        previous step. This allows reusing a previously written property in
        case that it does not change at different steps.
      step: Explicit ``int64``-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which
        must not be None.
      max_outputs: Optional ``int`` or rank-0 integer ``Tensor``. At most this
        many images will be emitted at each step. When more than
        `max_outputs` many images are provided, the first ``max_outputs`` many
        images will be used and the rest silently discarded.
      description: Optional long-form description for this summary, as a
        constant ``str``. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        raise ValueError("step is not provided or set.")

    summary_metadata = metadata.create_summary_metadata(description=description)
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (getattr(tf.summary.experimental, "summary_scope", None) or
                     tf.summary.summary_scope)
    with summary_scope(name, "open3d_summary", values=[data, max_outputs,
                                                       step]) as (tag, scope):
        # Defer preprocessing by passing it as a callable to write(),
        # wrapped in a LazyTensorCreator for backwards compatibility, so that we
        # only do this work when summaries are actually written, i.e. if
        # record_if() returns True.
        @lazy_tensor_creator.LazyTensorCreator
        def lazy_tensor():
            # FIXME: Use public API to get logdir
            from tensorflow.python.ops.summary_ops_v2 import _summary_state
            logdir = _summary_state.writer._metadata['logdir'].numpy().decode(
                'utf-8')
            write_dir = PluginDirectory(logdir, metadata.PLUGIN_NAME)
            geometry_metadata_string = _write_geometry_data(
                write_dir, tag, step, data, max_outputs)
            return tf.convert_to_tensor(geometry_metadata_string)

        return tf.summary.write(tag=tag,
                                tensor=lazy_tensor,
                                step=step,
                                metadata=summary_metadata)
