import open3d as o3d

filename = 'test_py.mkv'

# global flag
flag_exit = False
def escape_callback(vis):
    global flag_exit
    flag_exit = True
    return False

flag_pause = False
def space_callback(vis):
    global flag_pause
    flag_pause = not flag_pause
    return False

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(256, escape_callback)
vis.register_key_callback(32, space_callback)

azure_mkv_reader = o3d.io.AzureKinectMKVReader()
azure_mkv_reader.open(filename)
if not azure_mkv_reader.is_opened():
    print('Unable to open file {}'.format(filename))
    exit

vis_geometry_added = False
vis.create_window('reader', 1920, 540)
while not azure_mkv_reader.is_eof() and not flag_exit:
    if not flag_pause:
        rgbd = azure_mkv_reader.next_frame()
        if rgbd is None:
            continue

        if not vis_geometry_added:
            vis.add_geometry(rgbd)
            vis_geometry_added = True

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

azure_mkv_reader.close()
