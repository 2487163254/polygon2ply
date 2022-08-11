import os
import open3d as o3d
import numpy as np
from plane_detection import DetectMultiPlanes
from utils import DownSample, DrawResult, RemoveNoiseStatistical
import random
from tqdm import tqdm
import copy
import yaml
import time


DEBUG_MODE = True


def simplifypPcd(pcd, threshold_upper=0.5, threshold_lower=0.31):
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
    points = np.asarray(pcd.points)
    points = DownSample(points, voxel_size=0.003)
    points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

    #DrawPointCloud(points, color=(0.4, 0.4, 0.4))
    results = DetectMultiPlanes(
        points, min_ratio=0.2, threshold=0.03, iterations=3000)
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
        if filename == 'e8f89ac4-b060-11ec-9d25-7c10c921acb3.ply':
            break
        plt_dir = os.path.join(data_dir, filename)
        if DEBUG_MODE: print(plt_dir)
        pcd = o3d.io.read_point_cloud(plt_dir)
        # Search Up
        l = 0.31
        u = 0.5
        original_pcd = copy.copy(pcd)
        pcd = simplifypPcd(original_pcd, u, l)
        biggest_pcd = copy.copy(pcd)
        while len(pcd.points)!=0:
            l += 0.1
            u += 0.1
            pcd =  simplifypPcd(original_pcd, u, l)
            if len(biggest_pcd.points) < len(pcd.points):
                biggest_pcd = copy.copy(pcd)

        # Search Down
        l = 0.31
        u = 0.5
        pcd = simplifypPcd(original_pcd, u, l)
        while len(pcd.points)!=0:
            l -= 0.1
            u -= 0.1
            pcd =  simplifypPcd(original_pcd, u, l)
            if len(biggest_pcd.points) < len(pcd.points):
                biggest_pcd = copy.copy(pcd)

        planes, colors, plane_nums = findRoadPlane(biggest_pcd)
        if plane_nums == 1:
            goodlist.append(filename)
        elif plane_nums > 1:
            badlist.append(filename)
        else:
            needcheck.append(filename)
        if DEBUG_MODE: DrawResult(planes, colors)
    
    now = time.time()
    print('Elapsed Time: %d mins' % ((now - start)/60))
    with open('result.yaml', 'w') as f:
        timestr = time.ctime(now)
        dict = {'runtime':timestr, 'goodlist':goodlist, 'badlist':badlist}
        yaml.dump(dict, f)


if __name__ == "__main__":
    main()
