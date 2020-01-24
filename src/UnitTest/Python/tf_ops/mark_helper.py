import open3d as o3d
import pytest

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
tf_marks = [
    pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                       reason='tf ops not built'),
    pytest.mark.filterwarnings(
        'ignore::DeprecationWarning:.*(tensorflow|protobuf).*'),
]

# check for GPUs and set memory growth to prevent tf from allocating all memory
gpu_devices = []
try:
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in gpu_devices:
        tf.config.experimental.set_memory_growth(dev, True)
except:
    pass

# define the list of devices for running the ops
device_names = ['CPU:0']
if gpu_devices:
    device_names.append('GPU:0')
devices = pytest.mark.parametrize('device_name', device_names)
