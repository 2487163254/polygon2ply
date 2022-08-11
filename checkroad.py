import os
import open3d as o3d
import numpy as np
from plane_detection import DetectMultiPlanes
from utils import DownSample, DrawResult, RemoveNoiseStatistical
import time
import random


plt_dir = os.path.join(os.getcwd(), 'export', '0b09f56c-d692-11ec-b9e2-92640e831eb9.ply')
pcd = o3d.io.read_point_cloud(plt_dir)
np_points = np.asarray(pcd.points)
x = np_points[:,0]
y = np_points[:,1]
z = np_points[:,2]
# print(max(x))
# print(max(y))
# print(np_points.shape)
# print(max(z))

y_max = np.where(y <= 0.5)[0]
pcd = pcd.select_by_index(y_max)

np_points = np.asarray(pcd.points)
print(len(np_points))
x = np_points[:,0]
y = np_points[:,1]
z = np_points[:,2]
y_min = np.where(y >= 0.31)[0]
pcd = pcd.select_by_index(y_min)




points = np.asarray(pcd.points)
points = DownSample(points,voxel_size=0.003)
points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

#DrawPointCloud(points, color=(0.4, 0.4, 0.4))
t0 = time.time()
results = DetectMultiPlanes(points, min_ratio=0.1, threshold=0.015, iterations=3000)
print('Time:', time.time() - t0)
planes = []
colors = []
for [a, b, c, d], plane in results:

    r = random.random()
    g = random.random()
    b = random.random()
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0, with rgb={(r*255, g*255, b*255)}")

    color = np.zeros((plane.shape[0], plane.shape[1]))
    color[:, 0] = r
    color[:, 1] = g
    color[:, 2] = b

    planes.append(plane)
    colors.append(color)

planes = np.concatenate(planes, axis=0)
colors = np.concatenate(colors, axis=0)
DrawResult(planes, colors)


# np_points = np.asarray(pcd.points)
# print(len(np_points))

# y = np_points[:,1]

# y_min = np.where(y == min(y))
# y_max = np.where(y == max(y))
# print(min(y), max(y))

# np_colors = np.asarray(pcd.colors)
# np_colors[y_min] = (255, 0, 0)
# np_colors[y_max] = (255, 0, 255)
# pcd.colors = o3d.utility.Vector3dVector(np_colors)

# print(np_points[y_max], np_points[y_min])
# o3d.visualization.draw_geometries([pcd])
