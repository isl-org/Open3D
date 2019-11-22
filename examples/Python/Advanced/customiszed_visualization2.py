import open3d as o3d

mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

vis = o3d.visualization.VisualizerWithKeyCallback()

rotating = False

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)

def key_action_callback(vis, action, mods):
    print(action)
    global rotating
    if action == 1:  # key down
        rotating = True
    elif action == 0:  # key up
        rotating = False

vis.register_key_action_callback(32, key_action_callback)  # space

vis.create_window()
vis.add_geometry(mesh_box)

while True:
    if not vis.poll_events():
        break
    if rotating:
        rotate_view(vis)
    vis.update_renderer()

vis.destroy_window()