# %%
# import the necessary packages
import cv2
import open3d as o3d
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
# %% [markdown]
# # Project the centers and load the result back
# Now we use the control points lifted by reduced depth

# %%
import numpy as np
import open3d as o3d
import cv2
import os
import sys
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def gen_local_map_in_world2(depth_img, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=0, normal_img=None):
    # lift by the normal image
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]

    # local_pts_in_world = np.zeros((depth_img.shape[0] * depth_img.shape[1], 3))
    local_pts_in_world = np.empty((0, 3), float)

    # for i in range(depth_img.shape[0]): # row
    #
    #     for j in range(depth_img.shape[1]): # col
    
    if not (normal_img is None):
        normal_img = normal_img[:,:,[2,1,0]]
        # normal_img[:,:,0] = -normal_img[:,:,0]
        # normal_img[:,:,1] = -normal_img[:,:,1]
        # normal_img[:,:,2] = -normal_img[:,:,2]

    for center_id in range(super_pix_centers.shape[0]):

        i = super_pix_centers[center_id, 0] # row id
        j = super_pix_centers[center_id, 1] # col id

        depth = depth_img[i, j]

        if depth == 0:
            continue

        # print(str(depth))

        u = j # col
        v = i # row ?????????


        z = depth / depth_scale

        x = (u - cx) * z / fx

        y = (v - cy) * z / fy

        one_local_pt = [x, y, z]

        # print(str(x) + ' ' + str(y) + ' ' + str(z))

        one_local_pt = np.array(one_local_pt)
        
        if not (normal_img is None):
            nnorm = np.linalg.norm(normal_img[i,j,:])
            one_local_pt -= normal_img[i,j,:]/nnorm*(lift)
        
        one_local_pt = one_local_pt.reshape((3,1))

        t_c2w_ = t_c2w.reshape((3,1))

        pt_only_rot = np.matmul(R_c2w, one_local_pt)
        one_local_pt_in_world = pt_only_rot + t_c2w_

        one_local_pt_in_world = one_local_pt_in_world.reshape((1,3))
        
        # if not (normal_img is None):
        #     one_local_pt_in_world += normal_img[i,j,:]*(lift)

        # one_local_pt_in_world = one_local_pt_in_world.reshape(3)

        # local_pts_in_world[i*depth_img.shape[1] + j, :] = one_local_pt_in_world
        local_pts_in_world = np.append(local_pts_in_world, one_local_pt_in_world, axis=0)

        # local_pts.append(one_local_pt)


    return local_pts_in_world

def convert_depth_to_pcl(depth_image, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers=None, lift_depth=0.0, pts_per=1):
    
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    
    center_x = cx
    center_y = cy

    constant_x = 1 / fx
    constant_y = 1 / fy

    # pointcloud_xzyrgb_fields = [
    #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
    # ]

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    # Convert depth from mm to m.
    depth_image = depth_image / depth_scale - lift_depth

    x = np.multiply(depth_image, vs)
    y = depth_image * us[:, np.newaxis]

    stacked = np.ma.dstack((x, y, depth_image))
    if not (super_pix_centers is None):
        stacked = stacked[super_pix_centers[:,0],super_pix_centers[:,1]]
    compressed = stacked.compressed()
    
    
    pointcloud = compressed.reshape((int(compressed.shape[0] / 3), 3))

    # pointcloud = np.hstack((pointcloud[:, 0:3],
    #                         pack_bgr(*pointcloud.T[3:6])[:, None]))
    # pointcloud = [[point[0], point[1], point[2], point[3]]
    #               for point in pointcloud]

    # pointcloud = pc2.create_cloud(Header(), pointcloud_xzyrgb_fields,
    #                               pointcloud)
    
    H = np.eye(4,4)
    H[:3,:3] = R_c2w
    H[:3,3] = t_c2w
    
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(pointcloud)
    pt = pt.transform(H)
    pointcloud = np.asarray(pt.points)
    
    return pointcloud[::pts_per,:]

def convert_depth_to_pcl_normal(depth_image, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers=None, lift=0.0, pts_per=1,normal_image=None):
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    
    center_x = cx
    center_y = cy

    constant_x = 1 / fx
    constant_y = 1 / fy

    # pointcloud_xzyrgb_fields = [
    #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #     PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
    # ]

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    # Convert depth from mm to m.
    depth_image = depth_image / depth_scale

    x = np.multiply(depth_image, vs)
    y = depth_image * us[:, np.newaxis]

    stacked = np.ma.dstack((x, y, depth_image))

    if not(normal_image is None):
        norms = np.linalg.norm(normal_image,axis=2)
        norms = np.expand_dims(norms,axis=2)
        norms = np.repeat(norms,3,axis=2)
        ans = normal_image/norms
        stacked += ans*lift

    if not (super_pix_centers is None):
        stacked = stacked[super_pix_centers[:,0],super_pix_centers[:,1]]
    
    compressed = stacked.compressed()
    
    pointcloud = compressed.reshape((int(compressed.shape[0] / 3), 3))

    # pointcloud = np.hstack((pointcloud[:, 0:3],
    #                         pack_bgr(*pointcloud.T[3:6])[:, None]))
    # pointcloud = [[point[0], point[1], point[2], point[3]]
    #               for point in pointcloud]

    # pointcloud = pc2.create_cloud(Header(), pointcloud_xzyrgb_fields,
    #                               pointcloud)
    
    H = np.eye(4,4)
    H[:3,:3] = R_c2w
    H[:3,3] = t_c2w
    
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(pointcloud)
    pt = pt.transform(H)
    pointcloud = np.asarray(pt.points)
    
    return pointcloud[::pts_per,:]

def read_depthdata(depth_path):
    # load depth data and convert to 16UC1
    with open(depth_path,'rb') as f:
        data = np.fromfile(f,dtype=np.float32)
        data = data[2:] # try drop first 2
        depth = data.reshape([480,640])
    return depth
# plt.imshow(depth_root)

def visual_pts(pts,color=False,frame=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
    if color:
        pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6]/255)       
        
    list_visual = [pcd]
    if frame:
        frames = o3d.geometry.TriangleMesh.create_coordinate_frame()
        list_visual.append(frames)
    o3d.visualization.draw_geometries(list_visual)

def normal_transform(normals_c_oneView,R_c2w):
    a,b,c = normals_c_oneView.shape
    for i in range(a):
        for j in range(b):
            xyz = normals_c_oneView[i,j,:]
            xyz = np.matmul(R_c2w,xyz)
            normals_c_oneView[i,j,:] = xyz
    return normals_c_oneView
    
def lookat2quat(lookat_mat):
    origin = lookat_mat[:3]
    lookat_pts = lookat_mat[3:6]
    up_vec = lookat_mat[6:9]

    H = np.identity(4)

    camera_w = (lookat_pts-origin)/np.linalg.norm(lookat_pts-origin)

    camera_up = up_vec - np.dot(up_vec,camera_w)*camera_w
    camera_up = camera_up/np.linalg.norm(camera_up)

    camera_right = np.cross(camera_w,camera_up)
    camera_right = camera_right/np.linalg.norm(camera_right)

    H[:3,3] = origin
    H[:3,0] = camera_right
    H[:3,1] = -camera_up
    H[:3,2] = camera_w

    R_c2w = H[:3,:3]
    quat = R.from_matrix(R_c2w).as_quat()
    quat = np.array([quat[3],quat[0],quat[1],quat[2]]) # scalar first!!!!!
    return np.hstack([quat,origin])

def get_all_uvs(h,w):
    xs = np.array(list(range(0,h,10)))
    ys = np.array(list(range(0,w,10)))
    xxs, yys = np.meshgrid(xs,ys)
    return np.array([xxs.flatten(),yys.flatten()]).T
# %%
print(sys.path)

# scenes_root = "./data/formal_scenes/"
scenes_root = "./data/leitest3/"
# pose_root = "./data/poses_per2frame/every2frame/"
pose_root = "./data/leitest3pose/"
# output_root = "./output/lookats_control_point_normal/"
# output_root = "./output/lei_control_points3knormal/"
# output_root = "./output/masknolift_control_points/"
output_root = "./output/leitest3/"

# output_folder = './output_cam_file/'

# scene_names_path = '/storage/user/lhao/hjp/ws_superpixel/haoang_control_pts/scene_names_demo.txt' # !!!!!!!!!!!!!!!!!!!!!!! TO adjust
# center_fold = "testclusters/"
# center_fold = "leiclusters/"
center_fold = "downright_cluster/"

depth_fold = "depth/"
normal_fold = "normal/"

depth_scale = 1  # ??????????
# intrinsic = [577.591, 578.73, 318.905, 242.684] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# intrinsic = [222.22, 222.22, 160, 120]
intrinsic = [577.8705679012345,577.8705679012345,320,240]

# lift = 0.005 #!!!!!!
lift = 0.0

TEST_ALL_UVS = False 
USE_WORLD_CAM = True
SAVE_ORIGINS = True

scene_names = os.listdir(scenes_root)

# scene_names = ["scene0370_02"]
# scene_names = ["scene0704_01"]

# scene_idx = 0
for scene_idx, scene_name in enumerate(scene_names):
    # scene_name = scene_name.rstrip()
    
    scene_path = scenes_root + scene_name

    # img poses
    poses_path = pose_root + scene_name + '.txt'
    # poses_path = 'haoang_process_results/' + scene_name + '.txt'

    # with open(poses_path) as f_poses:
    #     poses = f_poses.readlines()
    poses = np.loadtxt(poses_path)

    output_folder = output_root + scene_name + '/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pts_in_world_all_mulView = np.empty((0, 9), float)
    
    mykey = lambda x:int(x.split(".")[0].split("_")[1])
    
    depth_all_names = sorted(os.listdir(os.path.join(scene_path, depth_fold)),key=mykey)
    center_all_names = sorted(os.listdir(os.path.join(scene_path, center_fold)),key=mykey)
    normal_all_names = sorted(os.listdir(os.path.join(scene_path, normal_fold)),key=mykey)
    
    depth_prefix = depth_all_names[0].split('_')[0]
    depth_format = depth_all_names[0].split('.')[-1]
    
    normal_prefix = normal_all_names[0].split('_')[0]

    # pts_in_world_all_mulView = np.ones((len(depth_all_names),3),float)
    if TEST_ALL_UVS:
        pts_all = np.empty((0,3))

    # center_all_names = ['bgrnormal_11.txt'] #!!!!!!
    for center_name in center_all_names:

        pose_idx = int(center_name.split(".")[0].split("_")[1])

        # depth_name = depth_all_names[pose_idx-1]
        depth_name = depth_prefix+("_{}.".format(pose_idx))+depth_format

        # normal_name = normal_all_names[pose_idx-1]
        normal_name = normal_prefix+("_{}.pfm".format(pose_idx))

        # break
        depth_path = os.path.join(os.path.join(scene_path, depth_fold, depth_name))
        
        center_path = os.path.join(os.path.join(scene_path, center_fold, center_name))
        
        normal_path = os.path.join(os.path.join(scene_path, normal_fold, normal_name))
        
        if depth_path.split(".")[-1]=="png":
            depth_all = cv2.imread(depth_path,cv2.CV_16UC1)/5000.0
        else:
            depth_all = read_depthdata(depth_path)

        pose = poses[pose_idx-1,:]
        print("center:{} with pose:{}".format(str(center_name),str(pose)))
        
        if TEST_ALL_UVS:
            # if (pose_idx != 10) and (pose_idx!=14): # !!!!
            #     continue
            print(pose)

        q_c2w = [pose[1], pose[2], pose[3], pose[0]]
        q_c2w = np.array(q_c2w)
        # q_c2w = q_c2w.reshape((1, 4))
        r_c2w = R.from_quat(q_c2w)
        R_c2w = r_c2w.as_matrix()

        t_c2w = pose[4:7]
        t_c2w = np.array(t_c2w)
        
        normals_c_oneView = cv2.imread(normal_path,cv2.IMREAD_UNCHANGED)
        normals_c_oneView = 2*normals_c_oneView-1
        # break
        super_pix_centers = np.loadtxt(center_path,dtype=np.int32)
        # break
        
        if TEST_ALL_UVS:
            pts_oneView = convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w)
            np.savetxt(os.path.join(output_folder,"pts_oneView.txt"),pts_oneView)
            pts_all = np.append(pts_all, pts_oneView,axis=0)
            # continue
        
        # origin_in_world_oneView = gen_local_map_in_world2(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=0, normal_img=normals_c_oneView)

        # origin_in_world_oneView = convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift_depth=lift)
        normals_c_oneView = normals_c_oneView[:,:,[2,1,0]]
        normals_c_oneView[:,:,1] = -normals_c_oneView[:,:,1]
        normals_c_oneView[:,:,2] = -normals_c_oneView[:,:,2]
        origin_in_world_oneView = convert_depth_to_pcl_normal(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=lift,normal_image=normals_c_oneView)
        # origin_in_world_oneView = convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers=super_pix_centers, lift_depth=lift,pts_per=100)
        np.savetxt(os.path.join(output_folder,"origin_in_world_oneView.txt"),origin_in_world_oneView)
        if USE_WORLD_CAM:
            lookat_in_world_oneView = origin_in_world_oneView + np.array([0,0,1])
            up_vectors_oneView = np.zeros_like(origin_in_world_oneView)
            up_vectors_oneView[:,1] = 1
        else:
            # lookat_in_world_oneView =  gen_local_map_in_world2(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=2*lift, normal_img=normals_c_oneView)
            lookat_in_world_oneView = convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift_depth=3*lift)
            up_vectors_oneView = np.zeros_like(lookat_in_world_oneView)
            up_vectors_oneView[:,2] = 1

        pts_in_world_all_oneView = np.hstack([origin_in_world_oneView, lookat_in_world_oneView, up_vectors_oneView])

        pts_in_world_all_mulView = np.append(pts_in_world_all_mulView, pts_in_world_all_oneView, axis=0)

        print('Complete pose ' + str(pose_idx) + '/' + str(len(depth_all_names)) + ', scene ' + str(scene_idx + 1) + '/' + str(len(scene_names)))

        # pose_idx = pose_idx + 1
        # break
    output_path_all_mulView = os.path.join(output_folder, scene_name+'_control_cam.txt')
    # break
    if TEST_ALL_UVS:
        np.savetxt(os.path.join(output_folder,scene_name+'test_all_uvs.txt'),pts_all, header='#x y z')
        # continue
    
    output_quats_mulView = np.zeros([len(pts_in_world_all_mulView),7])
    for id,lookat_mat in enumerate(pts_in_world_all_mulView):
        output_quats_mulView[id,:] = lookat2quat(lookat_mat)
    np.savetxt(os.path.join(output_folder, scene_name+'_control_cam_pose.txt'), output_quats_mulView, header='# qw qx qy qz x y z') #!!!!
    
    with open(output_path_all_mulView, 'w') as f_output_path_all_mulView:
        num = pts_in_world_all_mulView.shape[0]
        f_output_path_all_mulView.write( str(num) + '\n')
        for i in range(num):
            # if not i%10==0:
            #     continue
            f_output_path_all_mulView.write(str(pts_in_world_all_mulView[i][0]) + ' ' + str(pts_in_world_all_mulView[i][1]) + ' ' + str(
                pts_in_world_all_mulView[i][2]) + '\n')
            f_output_path_all_mulView.write(str(pts_in_world_all_mulView[i][3]) + ' ' + str(pts_in_world_all_mulView[i][4]) + ' ' + str(
                pts_in_world_all_mulView[i][5]) + '\n')
            f_output_path_all_mulView.write(str(pts_in_world_all_mulView[i][6]) + ' ' + str(pts_in_world_all_mulView[i][7]) + ' ' + str(
                pts_in_world_all_mulView[i][8]) + '\n')
    
    # np.savetxt(output_path_all_mulView, pts_in_world_all_mulView)
    # break
    print('Comelete Scene ' + str(scene_idx + 1) + '/' + str(len(scene_names)) )
    print('===================================================')

# %%
# np.savetxt("pts_all.txt",pts_all)
# pts_all = np.loadtxt("pts_all.txt")
if SAVE_ORIGINS:
    np.savetxt(output_folder + scene_name+"origins.txt", pts_in_world_all_mulView[:,:3])
    np.savetxt(output_folder + scene_name+"lookats.txt", pts_in_world_all_mulView[:,3:6])