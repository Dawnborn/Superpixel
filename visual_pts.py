import numpy as np
import open3d as o3d
import cv2
import os
import sys
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

path1 = "/storage/user/lhao/hjp/ws_superpixel/output/longtest_new2/scene0370_02/scene0370_02origins.txt"
pcd1 = o3d.io.read_point_cloud(path1,format="xyz",remove_nan_points=True, remove_infinite_points=True, print_progress=True)

# all_uvs = np.loadtxt("/home/hjp/Documents/research/test_pointcloud/scene0370_02test_all_uvs.xyz")

path2 = "/storage/user/lhao/hjp/ws_superpixel/output/longtest_new/scene0370_02/scene0370_02test_all_uvs.xyz"
pcd2 = o3d.io.read_point_cloud(path2,format="xyz",remove_nan_points=True, remove_infinite_points=True, print_progress=True)
pcd2 = pcd2.voxel_down_sample(voxel_size=0.1)



pcd1.paint_uniform_color([1, 0, 0])
pcd2.paint_uniform_color([0, 1, 0])

list_visual = [pcd1, pcd2]
# frames = o3d.geometry.TriangleMesh.create_coordinate_frame()
# list_visual.append(frames)
o3d.visualization.draw_geometries(list_visual)

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultUnlit'
mat.point_size = 1.0

draw = o3d.visualization.EV.draw

draw([{'name': 'pcd', 'geometry': list_visual, 'material': mat}], show_skybox=False)
