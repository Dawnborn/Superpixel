import sys
import os
import numpy as np
import cv2

def normalpfm2data(pfm):
    normals_c_oneView = 2*pfm-1
    normals = normals_c_oneView[:,:,[2,1,0]]
    # normals[:,:,0] = -normals[:,:,0]
    normals[:,:,1] = -normals[:,:,1]
    normals[:,:,2] = -normals[:,:,2]
    return normals

if __name__ == "__main__":
    scene_root = "/storage/user/lhao/hjp/ws_superpixel/data/leitest3"
    scene_lists = os.listdir(scene_root)
    for scene_name in sorted(scene_lists):
        scene_path = os.path.join(scene_root,scene_name)
        pfm_normal_paths = os.path.join(scene_path,"normal")
        data_normal_paths = os.path.join(scene_path,"normal_dat")
        if not os.path.exists(data_normal_paths):
            os.makedirs(data_normal_paths)
        for pfm_normal_path in sorted(os.listdir(pfm_normal_paths)):
            pfm = cv2.imread(os.path.join(pfm_normal_paths,pfm_normal_path), cv2.IMREAD_UNCHANGED)
            normals = normalpfm2data(pfm)
            data_normal_path = pfm_normal_path.replace(".pfm",".dat")
            normals.astype(np.single).tofile(os.path.join(data_normal_paths, data_normal_path))