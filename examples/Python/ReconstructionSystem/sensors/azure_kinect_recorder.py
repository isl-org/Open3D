import argparse
import datetime
import open3d as o3d
import os


kbd_escape = 256
flag_stop = False
def escape_callback(vis):
    global flag_stop, recorder
    flag_stop = True
    if recorder.is_record_created():
        print('Recording finished')
    else:
        print('Nothing has been recorded')
    return False


kbd_space = 32
flag_record = False
def space_callback(vis):
    global flag_record, recorder, filename
    if flag_record:
        print('Recording paused. '
              'Press [Space] to continue. '
              'Press [ESC] to save and exit')
        flag_record = False

    elif not recorder.is_record_created():
        if recorder.open_record(filename):
            print('Recording started. '
                  'Press [SPACE] to pause. '
                  'Press [ESC] to save and exit')
            flag_record = True

    else:
        print('Recording resumed, video may be discontinuous. '
              'Press [SPACE] to pause. '
              'Press [ESC] to save and exit')
        flag_record = True

    return False


def main(recorder, align_depth_to_color):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(kbd_escape, escape_callback)
    vis.register_key_callback(kbd_space, space_callback)

    vis_geometry_added = False
    vis.create_window('recorder', 1920, 540)

    while not flag_stop:
        rgbd = recorder.record_frame(flag_record, align_depth_to_color)
        if rgbd is None:
            continue

        if not vis_geometry_added:
            vis.add_geometry(rgbd)
            vis_geometry_added = True

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str,
                        help='input json kinect config')
    parser.add_argument('--output', type=str,
                        help='output mkv filename')
    parser.add_argument('--list', action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device', type=int, default=0,
                        help='input kinect device id')
    parser.add_argument('-a', '--align_depth_to_color', action='store_true',
                        help='enable align depth image to color')

    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    if args.output is not None:
        filename = args.output
    else:
        filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(
            date=datetime.datetime.now())
    print('Prepare writing to {}'.format(filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    azure_kinect_recorder = o3d.io.AzureKinectRecorder(config, device)
    rc = azure_kinect_recorder.init_sensor()
    if rc != 0:
        print('Failed to connect to sensor')
        exit()

    main(azure_kinect_recorder, args.align_depth_to_color)

    azure_kinect_recorder.close_record()
