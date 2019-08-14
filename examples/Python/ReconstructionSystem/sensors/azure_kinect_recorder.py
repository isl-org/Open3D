import open3d as o3d

# global flag
flag_exit = False
flag_record = False
recorder = o3d.io.AzureKinectRecorder(o3d.io.AzureKinectSensorConfig(), 0)
recorder.init_sensor()

filename = 'test_py.mkv'

def escape_callback(vis):
    global flag_exit, recorder
    flag_exit = True
    if recorder.is_record_created():
        print('Recording finished')
    else:
        print('Nothing has been recorded')
    return False


def space_callback(vis):
    global flag_record, recorder
    if flag_record:
        print('Recording paused. Press [Space] to continue. Press [ESC] to save and exit')
        flag_record = False

    elif not recorder.is_record_created():
        if recorder.open_record(filename):
            print('Recording started. Press [SPACE] to pause. Press [ESC] to save and exit')
            flag_record = True

    else:
        print('Recording resumed, video may be discontinuous. Press [SPACE] to pause. Press [ESC] to save and exit')
        flag_record = True

    return False


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(256, escape_callback)
vis.register_key_callback(32, space_callback)

vis_geometry_added = False
vis.create_window('recorder', 1920, 540)
while not flag_exit:
    rgbd = recorder.record_frame(flag_record, False)
    if rgbd is None:
        continue

    if not vis_geometry_added:
        vis.add_geometry(rgbd)
        vis_geometry_added = True

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

recorder.close_record()
