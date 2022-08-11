import copy
import os
import random
import time

import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

from utils import DetectMultiPlanes

DEBUG_MODE = False


def DrawResult(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def simplifypPcd(pcd, threshold_upper=0.5, threshold_lower=0.31):
    """simplifypPcd 取出两个阈值中间的pcd

    Args:
        pcd (o3d.PointCloud): 点云
        threshold_upper (float, optional): 上限. Defaults to 0.5.
        threshold_lower (float, optional): 下限. Defaults to 0.31.

    Returns:
        o3d.PointCloud: 点云
    """    
    np_points = np.asarray(pcd.points)
    y = np_points[:, 1]

    y_max = np.where(y <= threshold_upper)[0]
    pcd = pcd.select_by_index(y_max)

    np_points = np.asarray(pcd.points)

    y = np_points[:, 1]
    y_min = np.where(y >= threshold_lower)[0]
    pcd = pcd.select_by_index(y_min)

    return pcd


def findRoadPlane(pcd):
    """findRoadPlane 聚类找出有多少个平面

    Args:
        pcd (o3d.PointCloud): 点云

    Returns:
        list: pcd.potins, pcd.colors, plane_nums
    """    
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    results = DetectMultiPlanes(np.asarray(pcd.points), min_ratio=0.2, threshold=0.03, iterations=3000)
    planes = []
    colors = []
    plane_nums = len(results)
    for [a, b, c, d], plane in results:

        r = random.random()
        g = random.random()
        b = random.random()
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0, with rgb={(r*255, g*255, b*255)}")

        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b

        planes.append(plane)
        colors.append(color)

    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)
    return planes, colors, plane_nums


def main():
    start = time.time()
    root = os.getcwd()
    data_dir = os.path.join(root, 'debug')
    pcd_lists = os.listdir(data_dir)
    goodlist = []
    badlist = []
    needcheck = []
    for filename in tqdm(pcd_lists):
        plt_dir = os.path.join(data_dir, filename)
        if DEBUG_MODE:
            print(plt_dir)
        pcd = o3d.io.read_point_cloud(plt_dir)

        # Search Up
        l = 0.31
        u = 0.5
        original_pcd = copy.copy(pcd)
        pcd = simplifypPcd(original_pcd, u, l)
        biggest_pcd = copy.copy(pcd)
        while len(pcd.points) != 0:
            l += 0.1
            u += 0.1
            pcd = simplifypPcd(original_pcd, u, l)
            if len(biggest_pcd.points) < len(pcd.points):
                biggest_pcd = copy.copy(pcd)

        # Search Down
        l = 0.31
        u = 0.5
        pcd = simplifypPcd(original_pcd, u, l)
        while len(pcd.points) != 0:
            l -= 0.1
            u -= 0.1
            pcd = simplifypPcd(original_pcd, u, l)
            if len(biggest_pcd.points) < len(pcd.points):
                biggest_pcd = copy.copy(pcd)

        planes, colors, plane_nums = findRoadPlane(biggest_pcd)
        if plane_nums == 1:
            goodlist.append(filename)
        elif plane_nums > 1:
            badlist.append(filename)
        else:
            needcheck.append(filename)
        if DEBUG_MODE:
            DrawResult(planes, colors)

    now = time.time()
    print('Elapsed Time: %d mins' % ((now - start)/60))
    with open('result.yaml', 'w') as f:
        timestr = time.ctime(now)
        dict = {'runtime': timestr, 'goodlist': goodlist, 'badlist': badlist}
        yaml.dump(dict, f)


if __name__ == "__main__":
    main()
