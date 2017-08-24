import sys
sys.path.append("../..")
from py3d import *
from trajectory_io import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw

max_depth = 100000

def simple_rendering(X, Xc, h, w, intrinsic, extrinsic):
	img = np.ones([h,w,3])
	zbuffer = np.ones([h,w]) * max_depth
	R = extrinsic[:3,:3]
	t = extrinsic[:3,3]
	r = 1
	for xx, xc in zip(X, Xc):
		xx_trans = np.dot(R, xx.transpose()) + t
		xx_proj = np.dot(intrinsic, xx_trans)
		u_sub = xx_proj[0] / xx_proj[2]
		v_sub = xx_proj[1] / xx_proj[2]
		u = int(round(u_sub))
		v = int(round(v_sub))
		if u > 0 and u < w-1 and v > 0 and v < h-1 and zbuffer[v,u] > xx_trans[2]:
			#print([u,v])
			if r is not 0:
				img[v-r:v+r,u-r:u+r,:] = xc
				zbuffer[v-r:v+r,u-r:u+r] = xx_trans[2]
			else:
				img[v,u,:] = xc
				zbuffer[v,u] = xx_trans[2]
		#print([u,v])
	plt.imshow(img)
	plt.show()

def projection(X, Xc, h, w, intrinsic, extrinsic):
	img = np.zeros([h,w,3])
	zbuffer = np.ones([h,w]) * max_depth
	weighted_sum = np.zeros([h,w,3])
	weight_sum = np.zeros([h,w,3])
	R = extrinsic[:3,:3]
	t = extrinsic[:3,3]
	for xx, xc in zip(X, Xc):
		xx_trans = np.dot(R, xx.transpose()) + t
		xx_proj = np.dot(intrinsic, xx_trans)
		u_sub = xx_proj[0] / xx_proj[2]
		v_sub = xx_proj[1] / xx_proj[2]
		u = int(u_sub)
		v = int(v_sub)
		u_p = u_sub - u
		v_p = v_sub - v
		if u > 0 and u < w-1 and v > 0 and v < h-1 and zbuffer[v,u] > xx_trans[2]:
			#print([u,v])
			#img[v,u,:] = xc
			weighted_sum[v,u,:] = weighted_sum[v,u,:] + (1-v_p)*(1-u_p)*xc
			weighted_sum[v,u+1,:] = weighted_sum[v,u+1,:] + (1-v_p)*(u_p)*xc
			weighted_sum[v+1,u,:] = weighted_sum[v+1,u,:] + (v_p)*(1-u_p)*xc
			weighted_sum[v+1,u+1,:] = weighted_sum[v+1,u+1,:] + (v_p)*(u_p)*xc
			weight_sum[v,u,:] = weight_sum[v,u,:] + (1-v_p)*(1-u_p)
			weight_sum[v,u+1,:] = weight_sum[v,u+1,:] + (1-v_p)*(u_p)
			weight_sum[v+1,u,:] = weight_sum[v+1,u,:] + (v_p)*(1-u_p)
			weight_sum[v+1,u+1,:] = weight_sum[v+1,u+1,:] + (v_p)*(u_p)
			zbuffer[v,u] = xx_trans[2]
			zbuffer[v,u+1] = xx_trans[2]
			zbuffer[v+1,u] = xx_trans[2]
			zbuffer[v+1,u+1] = xx_trans[2]
		#print([u,v])
	for i in range(3):
		img[:,:,i] = weighted_sum[:,:,i] / (weight_sum[:,:,i] + 0.00000010)
	plt.imshow(img)
	plt.show()

def rendering(X, Xc, h, w, intrinsic, extrinsic):
	#img = np.zeros([h,w,3])
	img = Image.new('RGB', (w,h), (255,255,255))
	draw = ImageDraw.Draw(img)
	zbuffer = np.ones([h,w]) * max_depth
	R = extrinsic[:3,:3]
	t = extrinsic[:3,3]
	radius = int(1)
	for xx, xc in zip(X, Xc):
		xx_trans = np.dot(R, xx.transpose()) + t
		xx_proj = np.dot(intrinsic, xx_trans)
		u_sub = xx_proj[0] / xx_proj[2]
		v_sub = xx_proj[1] / xx_proj[2]
		u = int(round(u_sub))
		v = int(round(v_sub))
		if u >= radius and u < w-radius and v >= radius and v < h-radius and xx_trans[2] < zbuffer[v,u]:
			color = (int(xc[0]*255),int(xc[1]*255),int(xc[2]*255))
			draw.ellipse((u_sub - radius, v_sub - radius, u_sub + radius, v_sub + radius), fill=color)
			zbuffer[v-radius:v+radius,u-radius:u+radius] = xx_trans[2]
		#print([u,v])
	#for i in range(3):
	#	img[:,:,i] = weighted_sum[:,:,i] / (weight_sum[:,:,i] + 0.00000010)
	plt.imshow(img)
	plt.show()

if __name__ == "__main__":
	intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault
	extrinsic = read_trajectory("../../TestData/RGBD/odometry.log")
	#SetVerbosityLevel(VerbosityLevel.Debug)
	pcd = ReadPointCloud("../../TestData/ICP/cloud_bin_0.pcd")
	#pcd = VoxelDownSample(pcd_read, 0.01)
	height = 480
	width = 640
	#print(pcd)
	simple_rendering(np.asarray(pcd.points), np.asarray(pcd.colors), height, width,
			intrinsic.intrinsic_matrix, np.linalg.inv(extrinsic[0].pose))
