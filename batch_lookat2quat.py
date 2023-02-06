import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

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

if __name__ == "__main__":
    scene_root = "/storage/user/lhao/hjp/ws_superpixel/data/leitest3"
    scene_lists = os.listdir(scene_root)
    for scene_name in sorted(scene_lists):
        scene_path = os.path.join(scene_root,scene_name)
        raw = np.loadtxt(os.path.join(scene_path,"cam.txt"),skiprows=1)
        lookats = raw.reshape((-1,9))
        quatxyzs = np.zeros([len(lookats),7])
        for (id,lookat_mat) in enumerate(lookats):
            quatxyzs[id,:] = lookat2quat(lookat_mat)
        quatxyzs_path = os.path.join(scene_path,"cam_poses.txt")
        np.savetxt(quatxyzs_path,quatxyzs,header="qw qx qy qz x y z")