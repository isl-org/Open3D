import os, sys
import open3d as o3d
import numpy as np

from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_dataset

if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, jack_jack]',
               default='lounge')
    config = parser.get_config()

    depth_file_names, color_file_names = load_rgbd_file_names(config)

    os.makedirs(os.path.join(config.path_dataset, 'fragments'), exist_ok=True)

    frag_id = 0
    for i in range(0, len(depth_file_names), config.fragment_size):
        start = i
        end = min(i + config.fragment_size, len(depth_file_names))

        np.savetxt(os.path.join(config.path_dataset, 'fragments',
                                'fragment_{:03d}_colors.txt'.format(frag_id)),
                   color_file_names[start:end],
                   fmt='%s',
                   delimiter='')
        np.savetxt(os.path.join(config.path_dataset, 'fragments',
                                'fragment_{:03d}_depths.txt'.format(frag_id)),
                   depth_file_names[start:end],
                   fmt='%s',
                   delimiter='')

        frag_id += 1
