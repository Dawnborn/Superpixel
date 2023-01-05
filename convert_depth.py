import numpy as np
import os
# import imageio
import cv2
from tqdm import tqdm

def read_depthdata(depth_path):
    with open(depth_path,'rb') as f:
        data = np.fromfile(f,dtype=np.float32)
        data = data[2:] # try drop first 2
        depth = data.reshape([480,640])
    return depth

if __name__ == "__main__":

    print(os.getcwd())

    scene_path = "/storage/user/lhao/hjp/ws_superpixel/formal_scenes/scene0001_01/"

    depth_path = "depth/"

    new_depth_path = "newdepth/"

    depth_dirs = os.listdir(scene_path + depth_path)

    depth_dir = depth_dirs[0]

    for depth_dir in tqdm(depth_dirs):

        mypath = scene_path+depth_path+depth_dir

        depth = read_depthdata(mypath)

        # normal = img_as_float(io.imread(normal_root+scene_name+normal_dir))
        # plt.imshow(depth)

        # depth = cv2.Mat(depth/5000, cv2.CV_16UC1)
        depth = (depth*5000).astype(np.ushort) #uint16
        cv2.imwrite(scene_path+new_depth_path+depth_dir.split(".")[0]+".png",depth)

        # depth = (depth*5000).astype(np.ushort) #uint16
        # imageio.imwrite("test1.png",depth)