import tensorflow.compat.v2 as tf
from tensorboard.compat.proto import summary_pb2

from open3d.visualization.tensorboard_plugin import metadata
import ipdb


def greeting(name, guest, step=None, description=None):
    """Write a "greeting" summary.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      guest: A rank-0 string `Tensor`.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was written because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    summary_scope = (getattr(tf.summary.experimental, "summary_scope", None)
                     or tf.summary.summary_scope)
    with summary_scope(
            name,
            "greeting_summary",
            values=[guest, step],
    ) as (tag, _):
        return tf.summary.write(
            tag=tag,
            tensor=tf.strings.join(["Hello, ", guest, "!"]),
            step=step,
            metadata=_create_summary_metadata(description),
        )


def _create_summary_metadata(description):
    return summary_pb2.SummaryMetadata(
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=metadata.PLUGIN_NAME,
            content=b"",  # no need for summary-specific metadata
        ),
    )
