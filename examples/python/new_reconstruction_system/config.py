import json
from easydict import EasyDict as edict

# The properties are modifiable in 'default_config.json'.
# Note: True should be replaced with true in the json file.

default_json_content = {
    "name": "Default reconstruction system configuration",
    "path_dataset": "",
    "path_intrinsic": "",
    "fragment_size": 100,
    "device": "CUDA:0",
    "engine": "legacy",
    "engine_candidates": ["legacy", "tensor"],
    "multiprocessing": True,
    "preprocess": {
        "min_depth": 0.1,
        "max_depth": 3.0,
        "depth_scale": 1000.0
    },
    "odometry": {
        "method": "colored",
        "method_candidates": ["colored", "point2plane", "intensity", "hybrid"],
        "keyframe_interval": 5,
        "corres_distance_trunc": 0.07
    },
    "icp": {
        "method": "colored",
        "method_candidates": ["colored", "point2point", "point2plane"],
        "downsample_voxel_size": 0.05,
        "corres_distance_trunc": 0.07
    },
    "global_registration": {
        "method": "ransac",
        "method_candidates": ["fgr", "ransac"],
        "downsample_voxel_size": 0.05
    },
    "pose_graph": {
        "loop_closure_odometry_weight": 0.1,
        "loop_closure_registration_weight": 5.0
    },
    "integration": {
        "mode": "color",
        "mode_candidates": ["depth", "color"],
        "voxel_size": 0.0058,
        "sdf_trunc": 0.04,
        "scene_block_count": 40000,
        "fragment_block_count": 20000
    }
}

def recursive_print(d, offset=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print('{}{}:'.format(offset, k))
            recursive_print(v, offset+' '*4)
        else:
            print('{}{:35s} : {}'.format(offset, k, v))

# Mimic a class constructor with a function, using easydict as the underlying struct
def Config(file_name=None):
    config = None
    if file_name is None:
        config = edict(default_json_content)

    elif file_name.endswith('.json'):
        with open(file_name) as f:
            content = json.load(f)
            config = edict(content)

    else:
        print('Unsupported config file {}, abort'.format(file_name))

    # Resolve potential conflicts
    if config.engine == 'legacy':
        print('Legacy engine only supports CPU device.')
        config.device = 'CPU:0'
    if config.engine == 'tensor' and config.device == 'CUDA:0':
        print('Tensor engine with CUDA device does not support python multiprocessing.')
        config.multiprocessing = False

    return config

if __name__ == '__main__':
    config = Config()
    recursive_print(config)

