import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import open3d as o3d
import open3d.core as o3c
from config import ConfigParser

import numpy as np
import threading
import time
from common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh


class ReconstructionWindow:

    def __init__(self, config):
        self.config = config

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in fixed props
        fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Depth scale slider
        scale_label = gui.Label('Depth scale')
        scale_slider = gui.Slider(gui.Slider.INT)
        scale_slider.set_limits(1000, 5000)
        scale_slider.int_value = int(config.depth_scale)
        fixed_prop_grid.add_child(scale_label)
        fixed_prop_grid.add_child(scale_slider)

        voxel_size_label = gui.Label('Voxel size')
        voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
        voxel_size_slider.set_limits(3.0 / 512, 0.01)
        voxel_size_slider.double_value = config.voxel_size
        fixed_prop_grid.add_child(voxel_size_label)
        fixed_prop_grid.add_child(voxel_size_slider)

        est_block_count_label = gui.Label('Est. blocks')
        est_block_count_slider = gui.Slider(gui.Slider.INT)
        est_block_count_slider.set_limits(40000, 100000)
        voxel_size_slider.int_value = config.block_count
        fixed_prop_grid.add_child(est_block_count_label)
        fixed_prop_grid.add_child(est_block_count_slider)

        est_point_count_label = gui.Label('Est. points')
        est_point_count_slider = gui.Slider(gui.Slider.INT)
        est_point_count_slider.set_limits(500000, 1000000)
        fixed_prop_grid.add_child(est_point_count_label)
        fixed_prop_grid.add_child(est_point_count_slider)

        ## Items in adjustable props
        adjustable_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Reconstruction interval
        interval_label = gui.Label('Recon. interval')
        interval_slider = gui.Slider(gui.Slider.INT)
        interval_slider.set_limits(50, 500)
        adjustable_prop_grid.add_child(interval_label)
        adjustable_prop_grid.add_child(interval_slider)

        ### Depth max slider
        max_label = gui.Label('Depth max')
        max_slider = gui.Slider(gui.Slider.DOUBLE)
        max_slider.set_limits(3.0, 6.0)
        adjustable_prop_grid.add_child(max_label)
        adjustable_prop_grid.add_child(max_slider)

        ### Depth diff slider
        diff_label = gui.Label('Depth diff')
        diff_slider = gui.Slider(gui.Slider.DOUBLE)
        diff_slider.set_limits(0.07, 0.5)
        adjustable_prop_grid.add_child(diff_label)
        adjustable_prop_grid.add_child(diff_slider)

        ### Update surface?
        update_label = gui.Label('Update surface?')
        update_box = gui.Checkbox('')
        update_box.checked = True
        adjustable_prop_grid.add_child(update_label)
        adjustable_prop_grid.add_child(update_box)

        ### Ray cast color?
        raycast_label = gui.Label('Raycast color?')
        raycast_box = gui.Checkbox('')
        raycast_box.checked = True
        adjustable_prop_grid.add_child(raycast_label)
        adjustable_prop_grid.add_child(raycast_box)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.raycast_color_image = gui.ImageWidget()
        self.raycast_depth_image = gui.ImageWidget()
        tab2.add_child(self.raycast_color_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.raycast_depth_image)
        tabs.add_tab('Raycast images', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Starting settings'))
        self.panel.add_child(fixed_prop_grid)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(adjustable_prop_grid)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)

        self.is_done = False

        self.is_started = False
        self.is_running = False
        self.is_surface_updated = False

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    # Window layout callback
    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_start(self):
        max_points = 8000000
        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point['colors'] = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.Material()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        self.model = o3d.t.pipelines.slam.Model(self.config.voxel_size, 16,
                                                self.config.block_count,
                                                o3c.Tensor(np.eye(4)),
                                                o3c.Device(self.config.device))
        self.is_started = True

    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running

    def init_render(self, depth_ref, color_ref):
        self.input_depth_image.update_image(
            depth_ref.colorize_depth(config.depth_scale, config.depth_min,
                                     config.depth_max).to_legacy())
        self.input_color_image.update_image(color_ref.to_legacy())

        self.raycast_depth_image.update_image(
            depth_ref.colorize_depth(config.depth_scale, config.depth_min,
                                     config.depth_max).to_legacy())
        self.raycast_color_image.update_image(color_ref.to_legacy())
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        print(self.widget3d)

        # self.widget3d.look_at(center, center - (0, 1, 3), (0, -1, 0))

    def update_render(self, input_depth, input_color, raycast_depth,
                      raycast_color, pcd):
        self.input_depth_image.update_image(
            input_depth.colorize_depth(config.depth_scale, config.depth_min,
                                       config.depth_max).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())

        self.raycast_depth_image.update_image(
            raycast_depth.colorize_depth(config.depth_scale, config.depth_min,
                                         config.depth_max).to_legacy())
        self.raycast_color_image.update_image(raycast_color.to_legacy())
        if pcd is not None:
            self.widget3d.scene.scene.update_geometry(
                'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
                rendering.Scene.UPDATE_COLORS_FLAG)

    # Major loop
    def update_main(self):
        depth_file_names, color_file_names = load_rgbd_file_names(self.config)
        intrinsic = load_intrinsic(self.config)

        n_files = len(color_file_names)
        device = o3d.core.Device(config.device)

        T_frame_to_model = o3c.Tensor(np.identity(4))
        depth_ref = o3d.t.io.read_image(depth_file_names[0])
        color_ref = o3d.t.io.read_image(color_file_names[0])
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                 depth_ref.columns, intrinsic,
                                                 device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                   depth_ref.columns, intrinsic,
                                                   device)

        input_frame.set_data_from_image('depth', depth_ref)
        input_frame.set_data_from_image('color', color_ref)

        raycast_frame.set_data_from_image('depth', depth_ref)
        raycast_frame.set_data_from_image('color', color_ref)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render(depth_ref, color_ref))

        poses = []

        i = 0
        while not self.is_done:
            if not self.is_started or not self.is_running:
                time.sleep(0.05)
                continue

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            color = o3d.t.io.read_image(color_file_names[i]).to(device)

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)

            if i > 0:
                result = self.model.track_frame_to_model(
                    input_frame, raycast_frame, config.depth_scale,
                    config.depth_max, config.odometry_distance_thr)
                T_frame_to_model = T_frame_to_model @ result.transformation

            poses.append(T_frame_to_model.cpu().numpy())
            self.model.update_frame_pose(i, T_frame_to_model)
            self.model.integrate(input_frame, config.depth_scale,
                                 config.depth_max)
            self.model.synthesize_model_frame(raycast_frame, config.depth_scale,
                                              config.depth_min,
                                              config.depth_max, False)
            i += 1
            if i % 50 == 0:
                pcd = self.model.voxel_grid.extract_point_cloud().to(
                    o3d.core.Device('CPU:0'))
            else:
                pcd = None

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(
                    input_frame.get_data_as_image('depth'),
                    input_frame.get_data_as_image('color'),
                    raycast_frame.get_data_as_image('depth'),
                    raycast_frame.get_data_as_image('color'), pcd))


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()

    gui.Application.instance.initialize()
    w = ReconstructionWindow(config)
    gui.Application.instance.run()
