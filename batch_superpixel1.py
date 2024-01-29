# %%
# Selectiong 2D points by superpixel
# open3d37
#
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

# scene_root = "./data/formal_scenes/"
# scene_root = "/storage/user/lhao/hjp/ws_superpixel/data/cam_xi/"
# scene_root = "/storage/user/lhao/hjp/ws_superpixel/data/leitest3/"
# scene_root = "/storage/local/lhao/junpeng/chenglei_dataset/scene0370_02/nobunny/"
# scene_root = "/storage/local/lhao/junpeng/chenglei_dataset/scene0370_02"
# scene_root = "/storage/remote/atcremers95/lhao/junpeng/chenglei_dataset/"
# scene_root = "/storage/remote/atcremers95/lhao/junpeng/finetune_dataset"
scene_root = "/storage/user/lhao/hjp/ws_optix/assets/ttt/tt/Scenes/xml/000scene0370_02"
scene_root = "/home/wiss/lhao/storage/user/hjp/ws_optix/finetune_dataset"
scene_root = "/home/wiss/lhao/storage/user/hjp/ws_superpixel/ws_optix/assets/ttt/tt/Scenes/xml/000scene0594_00"

output_root = "/storage/user/lhao/hjp/ws_superpixel/output/"

hdr_fold = "/pfm/"
depth_fold = "/depth/"
normal_fold = "/normal/"

# cluster_fold = "normalclusters/"
# cluster_fold = "test200/"
# cluster_fold = "test200rgblei/"
# cluster_fold = "test2_150/" 
# cluster_fold = "/test2_300/" #### the output of control points, remember
# cluster_fold = "longtest_new3/"
# cluster_fold = "finetune_manual"
cluster_fold = "finetune_full"
cluster_fold = "finetune_full100"
cluster_fold = "manual650_200"
# cluster_fold = "normal_200"
cluster_fold = "normal_200step30"


segment_fold = cluster_fold+"_segments"


# %%
img_start = 0
img_end = 2000
# img_step = 50 #!!!!
img_step = 30
img_idx = list(range(img_start,img_end,img_step))
# img_idx = [0,36,63,88,98,152,179,238,282,298] # fine_tuned manual
# img_idx = [3]
# img_idx = [0,40,80,160,200,240,320,360,400,440,480,520,560,600,640]

numSegments = 200 #!!!!

folder = normal_fold # the folder use to cluster, change it to normal or basecolor or ldr for now
        
mykey = lambda x:int(x.split(".")[0].split("_")[1])

# scene_names = os.listdir(scene_root)
# scene_names = ['2frame0370_02']
# scene_names = ['300frame0370_02']
# scene_names = ['scene0551_00','scene0582_00','scene0594_00','scene0604_00']
# scene_names = ["60frame0370_02"]
scene_names = ["scene0582_00"]
scene_names = ["addon"]

for scene_name in scene_names:
    print("==================="+scene_name+"===================")
    # scene_name = scene_name + "/"
    
    img_dirs = sorted(os.listdir(os.path.join(scene_root,scene_name+folder)),key=mykey)
    
    img_end = min([img_end, len(img_dirs)])
    
    for id_img, img_dir in tqdm(enumerate(img_dirs)):
        
        # if id_img < img_start:
        #     continue
        
        # if id_img >= img_end:
        #     break      
        
        # if not id_img % img_step == 0:
        #     continue
        if not (id_img in img_idx):
            continue
        
        if img_dir.split(".")[-1]=="pfm":
            image = cv2.imread(os.path.join(scene_root,scene_name+folder,img_dir), cv2.IMREAD_UNCHANGED)
        else:
            image = img_as_float(io.imread(os.path.join(scene_root,scene_name,folder,img_dir)))

        segments = slic(image, n_segments = numSegments, sigma = 3, compactness = 1) ####### segments are per-pixel label!!!!!!
        # those look similar share the same envmap, either is really similar or they are robust to different envmaps

        numResult = len(set(segments.flatten()))
        centers = np.zeros([numResult,2])
        for s in range(segments.min(), segments.max()+1):
            centers[s-1,:]=np.argwhere(segments==s).mean(axis=0)
        centers = centers.round()
        img_result = mark_boundaries(image, segments)
        plt.imshow(img_result)
        plt.scatter(centers[:,1],centers[:,0],marker='o',color='r')
        cluster_path = os.path.join(scene_root, scene_name, cluster_fold)
        if not os.path.exists(cluster_path):
            os.mkdir(cluster_path)
        output_path = os.path.join(cluster_path, img_dir.split('.')[0]+".txt")

        print(str(centers.shape[0])+" centers writen to: "+output_path+"\n")

        vis_fold = os.path.join(output_root, cluster_fold, scene_name, "vis")
        if not os.path.exists(vis_fold):
            os.makedirs(vis_fold)
        vis_path = os.path.join(vis_fold,"vis_{}.png".format(str(img_dir.split('.')[0])))
        plt.savefig(vis_path)
        plt.cla()
        # break
        np.savetxt(output_path,centers,fmt='%d')
    
    # break
# %%
