import argparse
import datetime
import open3d as o3d
import os


flag_stop = False
def escape_callback(vis):
    global flag_stop, sensor
    flag_stop = True
    return False


def main(sensor, align_depth_to_color):
    glfw_key_escape = 256
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(glfw_key_escape, escape_callback)

    vis_geometry_added = False
    vis.create_window('viewer', 1920, 540)

    while not flag_stop:
        rgbd = sensor.capture_frame(align_depth_to_color)
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

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    sensor = o3d.io.AzureKinectSensor(config)
    rc = sensor.connect(device)
    if rc != 0:
        print('Failed to connect to sensor')
        exit()

    main(sensor, args.align_depth_to_color)
