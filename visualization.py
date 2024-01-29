#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# %%
raw_data = np.loadtxt("/storage/user/lhao/hjp/ws_superpixel/data/visualization/image_oriResidual.txt")
# plt.matshow(1000*raw_data)
#%%
raw_data2 = np.loadtxt("/storage/user/lhao/hjp/ws_superpixel/data/visualization/delta_comp_sparsed_useRatioComp.txt")
X = raw_data2[:,0]
Y = raw_data2[:,1]
Z1 = raw_data2[:,2]*1000
Z2 = raw_data2[:,3]*1000
# plt.show()


#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X, Y, Z1, c='red',s=0.1)
ax.scatter(X, Y, Z2, c='blue',s=0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.azim = 90
ax.dist = 10
ax.elev = 90

plt.show()

#%%
X1 = X[Z1>Z2]
Y1 = Y[Z1>Z2]
Z11 = Z1[Z1>Z2]
X2 = X[Z2>Z1]
Y2 = Y[Z2>Z1]
Z22 = Z2[Z2>Z1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Y1, X1, c='red', s=0.1)
ax.scatter(Y2, X2, c='blue', s=0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim([0,640])
ax.set_ylim([0,480])

plt.show()

#%%
ptc1 = raw_data2[:,[0,1,2]]
ptc2 = raw_data2[:,[0,1,3]]
ptc1[:,2] *= 1000
ptc2[:,2] *= 1000
np.savetxt('ptc1.txt',ptc1)
np.savetxt('ptc2.txt',ptc2)
#%%
cv2.imshow('raw',raw_data)