import numpy as np
import os

def ptc2lookat(origins_in_world):
    pass

if __name__ == "__main__":
    # ptc_path = "/storage/user/lhao/hjp/ws_superpixel/data/ptc_input/ControlpointCloud_Sparsfied_764.txt"
    ptc_path = "/storage/user/lhao/hjp/ws_superpixel/data/ptc_input/ControlpointCloud_Sparsfied_91.txt"
    output_folder = "/storage/user/lhao/hjp/ws_superpixel/output/ptc_input_output/"
    output_path = os.path.join(output_folder,"control_cam_"+ptc_path.split("/")[-1])
    output_pose_path = os.path.join(output_folder,"control_cam_pose_"+ptc_path.split("/")[-1])


    origins_in_world = np.loadtxt(ptc_path)
    lookats_in_world = origins_in_world + np.array([0,0,1])
    up_vectors_oneView = np.zeros_like(origins_in_world)
    up_vectors_oneView[:,1] = 1

    with open(output_path, 'w') as f_output_path_all_mulView:
        num = origins_in_world.shape[0]
        f_output_path_all_mulView.write( str(num) + '\n')
        for i in range(num):
            f_output_path_all_mulView.write(str(origins_in_world[i][0]) + ' ' + str(origins_in_world[i][1]) + ' ' + str(
                origins_in_world[i][2]) + '\n')
            f_output_path_all_mulView.write(str(lookats_in_world[i][0]) + ' ' + str(lookats_in_world[i][1]) + ' ' + str(
                lookats_in_world[i][2]) + '\n')
            f_output_path_all_mulView.write(str(up_vectors_oneView[i][0]) + ' ' + str(up_vectors_oneView[i][1]) + ' ' + str(
                up_vectors_oneView[i][2]) + '\n')

    quats = np.zeros([num,4])
    quats[:,3] = 1 # w x y z
    quatxyz = np.hstack((quats,origins_in_world))
    np.savetxt(output_pose_path,quatxyz,header="qw qx qy qz x y z")
