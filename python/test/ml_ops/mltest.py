import numpy as np
import open3d as o3d
import pytest
from collections import namedtuple
import importlib
from types import SimpleNamespace

# skip all tests if the ml ops were not built
default_marks = [
    pytest.mark.skipif(not (o3d._build_config['BUILD_TENSORFLOW_OPS'] or
                            o3d._build_config['BUILD_PYTORCH_OPS']),
                       reason='ml ops not built'),
]

MLModules = namedtuple('MLModules',
                       ['module', 'ops', 'layers', 'device', 'cpu_device'])

# define the list of frameworks and devices for running the ops
_ml_modules = {}
try:
    tf = importlib.import_module('tensorflow')
    ml3d_ops = importlib.import_module('open3d.ml.tf.ops')
    ml3d_layers = importlib.import_module('open3d.ml.tf.layers')
    _ml_modules['tf'] = MLModules(tf, ml3d_ops, ml3d_layers, 'CPU:0', 'CPU:0')
    # check for GPUs and set memory growth to prevent tf from allocating all memory
    tf_gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in tf_gpu_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    if tf_gpu_devices and o3d._build_config['BUILD_CUDA_MODULE']:
        _ml_modules['tf_gpu'] = MLModules(tf, ml3d_ops, ml3d_layers, 'GPU:0',
                                          'CPU:0')
except ImportError:
    pass

try:
    torch = importlib.import_module('torch')
    ml3d_ops = importlib.import_module('open3d.ml.torch.ops')
    ml3d_layers = importlib.import_module('open3d.ml.torch.layers')
    _ml_modules['torch'] = MLModules(torch, ml3d_ops, ml3d_layers, 'cpu', 'cpu')
    if torch.cuda.is_available() and o3d._build_config['BUILD_CUDA_MODULE']:
        _ml_modules['torch_cuda'] = MLModules(torch, ml3d_ops, ml3d_layers,
                                              'cuda', 'cpu')
except ImportError:
    pass


def is_gpu_device_name(name):
    return name in ('GPU:0', 'cuda')


def to_numpy(tensor):
    if 'torch' in _ml_modules and isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()

        return tensor.numpy()
    else:
        return tensor.numpy()


def to_torch(x, device):
    """Converts x such that it can be used as input to a pytorch op."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).contiguous().to(device)
    else:
        return x


def run_op(ml, device_name, check_device, fn, *args, **kwargs):
    """Runs an op using an ml framework"""
    if ml.module.__name__ == 'tensorflow':
        with tf.device(device_name):
            ans = fn(*args, **kwargs)

            if check_device:
                # not all returned tensors have to use the device.
                # check if there is at least one tensor using device memory
                tensor_on_device = False
                if isinstance(ans, tf.Tensor):
                    if device_name in ans.device:
                        tensor_on_device = True
                else:
                    for x in ans:
                        if device_name in x.device:
                            tensor_on_device = True
                assert tensor_on_device

    elif ml.module.__name__ == 'torch':
        _args = [to_torch(x, device_name) for x in args]
        _kwargs = {k: to_torch(v, device_name) for k, v in kwargs.items()}

        ans = fn(*_args, **_kwargs)

        if check_device:
            # not all returned tensor have to use the device.
            # check if there is at least one tensor using device memory
            tensor_on_device = False
            if isinstance(ans, torch.Tensor):
                if device_name == ans.device.type:
                    tensor_on_device = True
            else:
                for x in ans:
                    if isinstance(
                            x, torch.Tensor) and device_name == x.device.type:
                        tensor_on_device = True
            assert tensor_on_device

    else:
        raise ValueError('unsupported ml framework {}'.format(ml.module))

    # convert outputs to numpy.
    if hasattr(ans, 'numpy'):
        new_ans = to_numpy(ans)
    else:
        # we assume the output is a (named)tuple if there is no numpy() function
        return_type = type(ans)
        output_as_numpy = [to_numpy(x) for x in ans]
        new_ans = return_type(*output_as_numpy)

    return new_ans


def run_op_grad(ml, device_name, check_device, fn, x, y_attr_name,
                backprop_values, *args, **kwargs):
    """Computes the gradient for input x of an op using an ml framework"""
    if ml.module.__name__ == 'tensorflow':
        x_var = tf.constant(x)
        _args = [x_var if a is x else a for a in args]
        _kwargs = {k: x_var if a is x else a for k, a in kwargs.items()}
        with tf.device(device_name):
            with tf.GradientTape() as tape:
                tape.watch(x_var)
                ans = fn(*_args, **_kwargs)
                if y_attr_name:
                    y = getattr(ans, y_attr_name)
                else:
                    y = ans
                dy_dx = tape.gradient(y, x_var, backprop_values)

                if check_device:
                    # check if the gradient is using device memory
                    tensor_on_device = False
                    if device_name in dy_dx.device:
                        tensor_on_device = True
                    assert tensor_on_device
    elif ml.module.__name__ == 'torch':
        x_var = to_torch(x, device_name)
        x_var.requires_grad = True
        _args = [x_var if a is x else to_torch(a, device_name) for a in args]
        _kwargs = {
            k: x_var if a is x else to_torch(a, device_name)
            for k, a in kwargs.items()
        }

        ans = fn(*_args, **_kwargs)
        if y_attr_name:
            y = getattr(ans, y_attr_name)
        else:
            y = ans
        y.backward(to_torch(backprop_values, device_name))
        dy_dx = x_var.grad

        if check_device:
            # check if the gradient is using device memory
            tensor_on_device = False
            if isinstance(dy_dx,
                          torch.Tensor) and device_name == dy_dx.device.type:
                tensor_on_device = True
            assert tensor_on_device
    else:
        raise ValueError('unsupported ml framework {}'.format(ml.module))

    return to_numpy(dy_dx)


# add parameterizations for the ml module and the device
parametrize = SimpleNamespace(
    ml=pytest.mark.parametrize('ml', _ml_modules.values()),
    ml_cpu_only=pytest.mark.parametrize(
        'ml',
        [v for k, v in _ml_modules.items() if v.device in ('CPU:0', 'cpu')]),
    ml_gpu_only=pytest.mark.parametrize(
        'ml',
        [v for k, v in _ml_modules.items() if is_gpu_device_name(v.device)]),
)
