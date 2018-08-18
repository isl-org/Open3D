# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
from open3d import *

def run_precompute(config):
	p = param(config)
	make_dir(p.output_dir)

	for scan_name in list_dir(p.dataset_dir):
		for rot in range(0, p.num_rotations):
			cloud_file = os.path.join(p.dataset_dir, scan_name, "scan.pcd")
			label_file = os.path.join(p.dataset_dir, scan_name, "scan.labels")

			if not os.path.exists(cloud_file) or not os.path.exists(label_file):
				continue

			if p.noise_level != 0.0:
				print("processing scan: %s, rot: %d, noise: %f" %
						(scan_name, rot, p.noise_level))
			else:
				print("processing scan: %s, rot: %d" % (scan_name, rot))

			pcd_colors = read_point_cloud(cloud_file)
			pcd_labels = read_point_cloud(cloud_file)
			txt_labels = read_txt_labels(label_file)

			theta = rot * 2 * math.pi / p.num_rotations

			rot_matrix = np.asarray(
						   [[np.cos(theta), -np.sin(theta), 0],
							[np.sin(theta), np.cos(theta), 0],
							[0, 0, 1]])
			pcd_colors.points = Vector3dVector(np.matmul(np.asarray(pcd_colors.points),
											 			 rot_matrix))
			pcd_labels.points = Vector3dVector(np.matmul(np.asarray(pcd_labels.points),
											 			 rot_matrix))

			# additive gauissan noise
			if p.noise_level != 0.0:
				npts = len(pcd_colors.points)
				additive_noise_to_points = np.random.normal(0.0, p.noise_level, (npts,3))
				for i in range(npts):
					pcd_colors.points[i] += additive_noise_to_points[i]
					pcd_labels.points[i] += additive_noise_to_points[i]

			lb = np.repeat(np.expand_dims(txt_labels, axis=1), 3, axis=1)
			pcd_labels.colors = Vector3dVector(lb)

			min_bound = pcd_colors.get_min_bound() - p.min_cube_size * 0.5
			max_bound = pcd_colors.get_max_bound() + p.min_cube_size * 0.5

			make_dir(os.path.join(p.output_dir, scan_name))
			make_dir(os.path.join(p.output_dir, scan_name, str(rot)))

			for i in range(0, p.num_scales):
				multiplier = pow(2, i)
				pcd_colors_down = voxel_down_sample_for_surface_conv(pcd_colors, multiplier*p.min_cube_size,
					min_bound, max_bound, False)
				pcd_labels_down = voxel_down_sample_for_surface_conv(pcd_labels, multiplier*p.min_cube_size,
					min_bound, max_bound, True)

				if p.interp_method == "depth_densify_nearest_neighbor":
					method = depth_densify_nearest_neighbor
				elif p.interp_method == "depth_densify_gaussian_kernel":
					method = depth_densify_gaussian_kernel
				parametrization = planar_parametrization(pcd_colors_down.point_cloud,
							KDTreeSearchParamHybrid(radius = 2*multiplier*p.min_cube_size, max_nn=100),
							PlanarParameterizationOption(
							sigma = 1, number_of_neighbors=p.num_neighbors, half_patch_size=p.filter_size//2,
							depth_densify_method = method))

				num_points = np.shape(np.asarray(pcd_colors_down.point_cloud.points))[0]

				np.savez_compressed(os.path.join(p.output_dir, scan_name, str(rot), 'scale_' + str(i) + '.npz'),
						points=np.asarray(pcd_colors_down.point_cloud.points),
						colors=np.asarray(pcd_colors_down.point_cloud.colors),
						labels_gt=np.reshape(np.asarray(pcd_labels_down.point_cloud.colors)[:, 0], (num_points)),
						nn_conv_ind=parametrization.index[0],
						pool_ind=pcd_colors_down.cubic_id,
						depth=parametrization.depth.data)

				# TODO Add support for Gaussian kernel

				pcd_colors = pcd_colors_down.point_cloud
				pcd_labels = pcd_labels_down.point_cloud

			sys.stdout.flush()
		sys.stdout.write("\n")
