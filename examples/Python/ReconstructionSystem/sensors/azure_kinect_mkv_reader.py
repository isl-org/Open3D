import argparse
import open3d as o3d
import os
import json
import sys

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(pwd, '..'))
from initialize_config import initialize_config

flag_stop = False
flag_play = True


def escape_callback(vis):
    global flag_stop
    flag_stop = True
    return False


def space_callback(vis):
    global flag_play
    if flag_play:
        print('Playback paused, press [SPACE] to continue.')
    else:
        print('Playback resumed, press [SPACE] to pause.')

    flag_play = not flag_play
    return False


def main(reader, output_path):
    glfw_key_escape = 256
    glfw_key_space = 32
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(glfw_key_escape, escape_callback)
    vis.register_key_callback(glfw_key_space, space_callback)

    vis_geometry_added = False
    vis.create_window('reader', 1920, 540)

    print("MKV reader initialized. Press [SPACE] to start, [ESC] to exit.")

    if output_path is not None:
        abspath = os.path.abspath(output_path)
        metadata = reader.get_metadata()
        o3d.io.write_azure_kinect_mkv_metadata(
            '{}/intrinsic.json'.format(abspath), metadata)

        config = {
            'path_dataset': abspath,
            'path_intrinsic': '{}/intrinsic.json'.format(abspath)
        }
        initialize_config(config)
        with open('{}/config.json'.format(abspath), 'w') as f:
            json.dump(config, f, indent=4)

    idx = 0
    while not reader.is_eof() and not flag_stop:
        if flag_play:
            rgbd = reader.next_frame()
            if rgbd is None:
                continue

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            if output_path is not None:
                color_filename = '{0}/color/{1:05d}.jpg'.format(
                    output_path, idx)
                print('Writing to {}'.format(color_filename))
                o3d.io.write_image(color_filename, rgbd.color)

                depth_filename = '{0}/depth/{1:05d}.png'.format(
                    output_path, idx)
                print('Writing to {}'.format(depth_filename))
                o3d.io.write_image(depth_filename, rgbd.depth)
                idx += 1

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv reader.')
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='input mkv file')
    parser.add_argument('--output',
                        type=str,
                        help='output path to store color/ and depth/ images')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        exit()

    if args.output is None:
        print('No output path, only play mkv')
    elif os.path.isdir(args.output):
        print('Output path {} already existing, only play mkv'.format(
            args.output))
        args.output = None
    else:
        try:
            os.mkdir(args.output)
            os.mkdir('{}/color'.format(args.output))
            os.mkdir('{}/depth'.format(args.output))
        except (PermissionError, FileExistsError):
            print('Unable to mkdir {}, only play mkv'.format(args.output))
            args.output = None

    azure_mkv_reader = o3d.io.AzureKinectMKVReader()
    azure_mkv_reader.open(args.input)
    if not azure_mkv_reader.is_opened():
        print('Unable to open file {}'.format(args.input))
        exit()

    main(azure_mkv_reader, args.output)

    azure_mkv_reader.close()
