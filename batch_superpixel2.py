# %%
# Convert 2D points to 3D points
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
# scenes_root = "./data/leitest3/"
# scenes_root = "./data/gaoming_dataset"
# scenes_root = "./data/cam_xi/"
# scenes_root = "/storage/local/lhao/junpeng/chenglei_dataset/scene0370_02/nobunny/"
# scenes_root = "/storage/remote/atcremers95/lhao/junpeng/chenglei_dataset/"
# scenes_root = "/storage/remote/atcremers95/lhao/junpeng/finetune_dataset/"
# scenes_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0370_02/"
# scenes_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/finetune_dataset/"
# scenes_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0594_00/"
# scenes_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0604_00"
scenes_root = "/storage/user/lhao/hjp/ws_superpixel/ws_optix/assets/ttt/tt/Scenes/xml/000scene0643_00"
scenes_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0086_00"

pose_root = "./data/poses_per2frame/every2frame/"
# pose_root = "./data/leitest3pose/"
# pose_root = "./data/cam_xi_poses/"

# output_root = "./output/lookats_control_point_normal/"
# output_root = "./output/lei_control_points3knormal/"
# output_root = "./output/masknolift_control_points/"
# output_root = "./output/test200/"
# output_root = "./output/test200plus/"
# output_root = "./output/test200pluslift/"
# output_root = "./output/test200pluslift5mm/"
# output_root = "./output/gaoming_dataset/"
# output_root = "./output/test2_300/"
# output_root = "./output/longtest_new/
# output_root = "./output/longtest_new2/"
# output_root = "./output/longtest_new3/"
output_root = "./output/finetune_manual/"
output_root = "./output/finetune_full/"
output_root = "./output/finetune_full100/"
output_root = "./output/manual650/"
output_root = "./output/manual650_200"
output_root = "./output/manual650_200step30"
output_root = "./output/manual650_100step30"
# output_folder = './output_cam_file/'

# scene_names_path = '/storage/user/lhao/hjp/ws_superpixel/haoang_control_pts/scene_names_demo.txt' # !!!!!!!!!!!!!!!!!!!!!!! TO adjust
# center_fold = "testclusters/"
# center_fold = "leiclusters/"
# center_fold = "downright_cluster/"
# center_fold = "test200rgblei/"
# center_fold = "longtest_new/"
# center_fold = "longtest_new2/"
center_fold = "longtest_new3/"
center_fold = "finetune_manual"
center_fold = "finetune_full"
center_fold = "finetune_full100"
center_fold = "manual650"
center_fold = "manual650_200"
center_fold = "normal_200"
center_fold = "normal_200step30"
center_fold = "normal_100step30"


depth_fold = "depth/"
normal_fold = "normal/"

depth_scale = 1  # ??????????
# intrinsic = [577.591, 578.73, 318.905, 242.684] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# intrinsic = [222.22, 222.22, 160, 120]
# intrinsic = [574.540648625183,574.540648625183,320,240] # scene0307_02
intrinsic = [577.8705679012345,577.8705679012345,320,240]

# lift = 0.005 #!!!!!!
# lift = 0.005
lift = 0.005

TEST_ALL_UVS = True 
USE_WORLD_CAM = True # all cameras are set to the same orientation, as the world frame
SAVE_ORIGINS = True

scene_names = os.listdir(scenes_root)

# scene_names = ["scene0370_02"]
# scene_names = ["scene0704_01"]
# scene_names = ["scene0066_00"]
# scene_names = ["300frame0370_02"]
# scene_names = ['scene0551_00','scene0582_00','scene0594_00','scene0604_00']
# scene_names = ["60frame0370_02"]
# scene_names = ["scene0582_00"]
# scene_names = ["addon"]
# scene_names = ["xi"]
scene_names = ["cam_manual_interpolated"]

# scene_idx = 0
for scene_idx, scene_name in enumerate(scene_names):
    # scene_name = scene_name.rstrip()
    
    scene_path = os.path.join(scenes_root, scene_name)

    # img poses
    # poses_path = os.path.join(pose_root, scene_name + '.txt')
    # poses_path = os.path.join(scene_path, "scene0604_00_xi_poses_formatted.txt")
    poses_path = "/storage/user/lhao/hjp/ws_superpixel/ws_optix/assets/ttt/tt/Scenes/xml/000scene0643_00/cam_manual_interpolated_poses_formatted.txt"
    poses_path = "/home/wiss/lhao/storage/user/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0086_00/cam_manual_interpolated/cam_manual_interpolated_poses_formatted.txt"

    if not (os.path.exists(poses_path)):
        import pdb
        pdb.set_trace()
    # poses_path = 'haoang_process_results/' + scene_name + '.txt'

    # with open(poses_path) as f_poses:
    #     poses = f_poses.readlines()
    poses = np.loadtxt(poses_path)

    output_folder = os.path.join(output_root, scene_name)
    # output_folder = scene_path
    
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
            # np.savetxt(os.path.join(output_folder,"pts_oneView.txt"),pts_oneView)
            pts_oneView = pts_oneView[::10,:]
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
    control_cam_pose_path = os.path.join(output_folder, scene_name+'_control_cam_pose.txt')
    np.savetxt(control_cam_pose_path, output_quats_mulView, header='# qw qx qy qz x y z') #!!!!
    print("control_cam_pose is written to {}!".format(control_cam_pose_path))
    
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
    print("control_cam is written to {}!".format(output_path_all_mulView))
    
    # np.savetxt(output_path_all_mulView, pts_in_world_all_mulView)
    # break
    print('Comelete Scene ' + str(scene_idx + 1) + '/' + str(len(scene_names)) )
    print('===================================================')

# %%
# np.savetxt("pts_all.txt",pts_all)
# pts_all = np.loadtxt("pts_all.txt")
if SAVE_ORIGINS:
    np.savetxt(os.path.join(output_folder, scene_name+"origins.xyz"), pts_in_world_all_mulView[:,:3])
    np.savetxt(os.path.join(output_folder, scene_name+"lookats.xyz"), pts_in_world_all_mulView[:,3:6])
# %%
