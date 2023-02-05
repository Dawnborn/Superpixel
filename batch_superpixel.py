# %%
# import the necessary packages
from skimage.segmentation import slic,watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import cv2
import open3d as o3d
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import sys

from tqdm import tqdm

# %%
print("#########################Generate superpixels")

scene_root = "./data/formal_scenes/"
hdr_fold = "hdr/"
depth_fold = "depth/"
normal_fold = "normal/"
cluster_fold = "normalclusters/"

# %%
img_start = 0
img_end = 1000
img_step = 50 #!!!!

numSegments = 200 #!!!!

folder = "normal/" # change to normal or basecolor or ldr for now
        
mykey = lambda x:int(x.split(".")[0].split("_")[1])

scene_names = os.listdir(scene_root)

for scene_name in scene_names:
    print("==================="+scene_name+"===================")
    scene_name = scene_name + "/"
    
    img_dirs = sorted(os.listdir(scene_root+scene_name+folder),key=mykey)
    
    img_end = min([img_end, len(img_dirs)])
    
    for id_img, img_dir in tqdm(enumerate(img_dirs)):
        
        if id_img < img_start:
            continue
        
        if id_img >= img_end:
            break      
        
        if not id_img % img_step == 0:
            continue
        
        if img_dir.split(".")[-1]=="pfm":
            image = cv2.imread(scene_root+scene_name+folder+img_dir, cv2.IMREAD_UNCHANGED)
        else:
            image = img_as_float(io.imread(scene_root+scene_name+folder+img_dir))

        segments = slic(image, n_segments = numSegments, sigma = 3, compactness=1, max_size_factor=1)
        # those look similar share the same envmap, either is really similar or they are robust to different envmaps
        
        numResult = len(set(segments.flatten()))
        centers = np.zeros([numResult,2])
        for s in range(segments.min(), segments.max()+1):
            centers[s-1,:]=np.argwhere(segments==s).mean(axis=0)
        centers = centers.round()
        # plt.imshow(mark_boundaries(image, segments))
        
        plt.scatter(centers[:,1],centers[:,0],marker='o',color='r')
        cluster_path = scene_root + scene_name + cluster_fold
        if not os.path.exists(cluster_path):
            os.mkdir(cluster_path)
        output_path = cluster_path+img_dir.split('.')[0]+".txt"
        print(str(centers.shape[0])+" centers writen to: "+output_path+"\n")
        # break
        np.savetxt(output_path,centers,fmt='%d')
    
    # break

# %% [markdown]
print("#########################Project the centers and load the result back...")
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

# %%
print(sys.path)

scenes_root = "./data/formal_scenes/"
pose_root = "./data/poses_per2frame/every2frame/"
output_root = "./output/lookats_control_point_normal/"

# output_folder = './output_cam_file/'

# scene_names_path = '/storage/user/lhao/hjp/ws_superpixel/haoang_control_pts/scene_names_demo.txt' # !!!!!!!!!!!!!!!!!!!!!!! TO adjust
center_fold = "normalclusters/"

depth_fold = "depth/"
normal_fold = "normal/"

depth_scale = 1  # ??????????

intrinsic = [577.8705679012345,577.8705679012345,320,240]

lift = 0.005 #!!!!!!

TEST_ALL_UVS = False

scene_names = os.listdir(scenes_root)

# scene_idx = 0
for scene_idx, scene_name in enumerate(scene_names):

    scene_path = scenes_root + scene_name

    poses_path = pose_root + scene_name + '.txt'
 
    poses = np.loadtxt(poses_path)

    output_folder = output_root + scene_name + '/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pts_in_world_all_mulView = np.empty((0, 9), float)
    
    mykey = lambda x:int(x.split(".")[0].split("_")[1])
    
    depth_all_names = sorted(os.listdir(os.path.join(scene_path, depth_fold)),key=mykey)
    center_all_names = sorted(os.listdir(os.path.join(scene_path, center_fold)),key=mykey)
    normal_all_names = sorted(os.listdir(os.path.join(scene_path, normal_fold)),key=mykey)

    # pts_in_world_all_mulView = np.ones((len(depth_all_names),3),float)
    if TEST_ALL_UVS:
        pts_all = np.empty((0,3))

    for center_name in center_all_names:

        pose_idx = int(center_name.split(".")[0].split("_")[1])

        depth_name = depth_all_names[pose_idx-1]

        normal_name = normal_all_names[pose_idx-1]

        # break
        depth_path = os.path.join(os.path.join(scene_path, depth_fold, depth_name))
        
        center_path = os.path.join(os.path.join(scene_path, center_fold, center_name))
        
        normal_path = os.path.join(os.path.join(scene_path, normal_fold, normal_name))
        
        depth_all = read_depthdata(depth_path)
        
        # pose = poses[pose_idx].rstrip()
        # pose = pose.split(' ')
        # pose = list(map(float, pose))

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
        
        normals_c_oneView = plt.imread(normal_path)
        # break
        super_pix_centers = np.loadtxt(center_path,dtype=np.int32)
        # break
        
        if TEST_ALL_UVS:
            pts_oneView = convert_depth_to_pcl(depth_all,intrinsic, depth_scale, R_c2w, t_c2w)
            pts_all = np.append(pts_all,pts_oneView,axis=0)
            continue
        
        # origin_in_world_oneView = gen_local_map_in_world2(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=0, normal_img=normals_c_oneView)
        origin_in_world_oneView = convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift_depth=lift)
        
        # lookat_in_world_oneView =  gen_local_map_in_world2(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift=2*lift, normal_img=normals_c_oneView)
        lookat_in_world_oneView =  convert_depth_to_pcl(depth_all, intrinsic, depth_scale, R_c2w, t_c2w, super_pix_centers, lift_depth=3*lift)
        
        up_vectors_oneView = np.zeros_like(lookat_in_world_oneView)
        up_vectors_oneView[:,2] = 1

        pts_in_world_all_oneView = np.hstack([origin_in_world_oneView, lookat_in_world_oneView, up_vectors_oneView])

        pts_in_world_all_mulView = np.append(pts_in_world_all_mulView, pts_in_world_all_oneView, axis=0)

        print('Complete pose ' + str(pose_idx) + '/' + str(len(depth_all_names)) + ', scene ' + str(scene_idx + 1) + '/' + str(len(scene_names)))

        # pose_idx = pose_idx + 1
        # break
    output_path_all_mulView = output_folder + scene_name + '_control_cam' +'.txt'
    # break
    if TEST_ALL_UVS:
        np.savetxt(output_folder + scene_name + 'test_all_uvs' +'.txt',pts_all, header='#x y z')
        continue
    
    output_quats_mulView = np.zeros([len(pts_in_world_all_mulView),7])
    for id,lookat_mat in enumerate(pts_in_world_all_mulView):
        output_quats_mulView[id,:] = lookat2quat(lookat_mat)
    np.savetxt(output_folder + scene_name + '_control_cam_pose' +'.txt',output_quats_mulView, header='# qw qx qy qz x y z') #!!!!
    
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
