import numpy as np
import open3d as o3d
import pytest
from collections import namedtuple
import importlib

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
tf_marks = [
    pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                       reason='tf ops not built'),
    pytest.mark.filterwarnings(
        'ignore::DeprecationWarning:.*(tensorflow|protobuf).*'),
]

MLModules = namedtuple('MLModules', ['framework', 'ops'])

# define the list of devices for running the ops
device_names = set(['CPU:0'])
_ml_modules = {}
try:
    tf = importlib.import_module('tensorflow')
    ml3d_ops = importlib.import_module('open3d.ml.tf.ops')
    _ml_modules['tf'] = MLModules(tf, ml3d_ops)
    # check for GPUs and set memory growth to prevent tf from allocating all memory
    tf_gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in tf_gpu_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    if tf_gpu_devices: device_names.add(['GPU:0'])
except:
    pass

try:
    torch = importlib.import_module('torch')
    ml3d_ops = importlib.import_module('open3d.ml.torch.nn.functional')
    _ml_modules['torch'] = MLModules(torch, ml3d_ops)
    if torch.cuda.is_available(): device_names.add(['GPU:0'])
except:
    pass


def to_numpy(tensor):
    if 'torch' in _ml_modules and isinstance(
            tensor, torch.Tensor) and tensor.device.type == 'cuda':
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def to_torch(x, device):
    """Converts x such that it can be used as input to a pytorch op."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).contiguous().to(device)
    else:
        return x


def run_op(ml_framework, device_name, fn, check_device, *args, **kwargs):
    if ml_framework.__name__ == 'tensorflow':
        with tf.device(device_name):
            ans = fn(*args, **kwargs)

            if check_device:
                # not all returned tensor have to use the device.
                # check if there is at least one tensor using device memory
                tensor_on_device = False
                for x in ans:
                    if device_name in x.device:
                        tensor_on_device = True
                assert tensor_on_device

    elif ml_framework.__name__ == 'torch':
        if 'GPU' in device_name:
            device = 'cuda'
        else:
            device = 'cpu'
        _args = [to_torch(x, device) for x in args]
        _kwargs = {k: to_torch(v, device) for k, v in kwargs.items()}

        ans = fn(*_args, **_kwargs)

        if check_device:
            # not all returned tensor have to use the device.
            # check if there is at least one tensor using device memory
            tensor_on_device = False
            for x in ans:
                if isinstance(x, torch.Tensor) and device == x.device.type:
                    tensor_on_device = True
            assert tensor_on_device

    else:
        raise ValueError('unsupported ml_framework {}'.format(ml_framework))

    # convert outputs to numpy. we assume the output is a (named)tuple
    return_type = type(ans)
    output_as_numpy = [to_numpy(x) for x in ans]
    new_ans = return_type(*output_as_numpy)
    return new_ans


ml = pytest.mark.parametrize('ml', _ml_modules.values())
devices = pytest.mark.parametrize('device_name', device_names)
