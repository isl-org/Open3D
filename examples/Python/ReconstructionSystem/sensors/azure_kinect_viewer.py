import open3d as o3d

# global flag
flag_exit = False
def escape_callback(vis):
    global flag_exit
    flag_exit = True
    return False


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(256, escape_callback)

sensor = o3d.io.AzureKinectSensor(o3d.io.AzureKinectSensorConfig())
sensor.connect(0)

vis_geometry_added = False
vis.create_window('recorder', 1920, 540)
while not flag_exit:
    rgbd = sensor.capture_frame(True)
    if rgbd is None:
        continue

    if not vis_geometry_added:
        vis.add_geometry(rgbd)
        vis_geometry_added = True

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
