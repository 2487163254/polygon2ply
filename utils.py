import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

ROOT_DIR = os.getcwd()
left_img_root = os.path.join(ROOT_DIR, "img")
disp_img_root = os.path.join(ROOT_DIR, "disp")
save_path = os.path.join(ROOT_DIR, "export")
label_nums = {}

color_list = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255), (221, 160, 221) ]

Zed_fx = 1067.27099
Zed_cx = 983.2330932617188
Zed_cy = 526.3822631835938
Zed_baseline = 119.731
Zed_w = 1920


def dep_to_pts_and_mask(depth_img, polygon, fx, cx, cy):
    # dont save max distance in plt, max distance 30 meter
    depth_img[polygon != 255] = 0
    depth_img[depth_img > (30 * 1000.)] = 0
    # print(get_mean_depth(depth_img))
    # print(depth_img[depth_img.nonzero()])
    # sns.boxplot(x=depth_img[depth_img.nonzero()])

    v, u = np.nonzero(depth_img)
    z = depth_img[v, u]
    z /= 1000.

    x = (u - cx) * z / fx
    y = (v - cy) * z / fx  # fy is the same as fx
    return x, y, z


def get_mean_depth(depth_img):
    non_zero = (depth_img != 0)
    return depth_img.sum() / non_zero.sum()



def dep_to_pts(depth_img, rgb_img, fx, cx, cy):
    # dont save max distance in plt, max distance 30 meter
    depth_img[depth_img > (30 * 1000.)] = 0
    v, u = np.nonzero(depth_img)
    z = depth_img[v, u]
    z /= 1000.

    x = (u - cx) * z / fx
    y = (v - cy) * z / fx  # fy is the same as fx

    color = rgb_img[v, u]
    return x, y, z, color


def save_plt(s_path, depth_img, polygon, color, fx, cx, cy):
    x, y, z = dep_to_pts_and_mask(depth_img, polygon, fx, cx, cy)

    points = []
    for X, Y, Z in zip(x, y, z):
        points.append("%f %f %f %d %d %d 0\n" %
                      (X, Y, Z, color[0], color[1], color[2]))
    file = open(s_path, "w")
    file.write('''ply
              format ascii 1.0
              element vertex %d
              property float x
              property float y
              property float z
              property uchar red
              property uchar green
              property uchar blue
              property uchar alpha
              end_header
              %s
              ''' % (len(points), "".join(points)))
    file.close()


def get_ply(img_name, mask, label, color, s_plt_path):
    left_img_path = os.path.join(left_img_root, img_name+'.png')
    left_img = cv2.imread(left_img_path)
    left_img = cv2.resize(
        left_img, (int(left_img.shape[1]/4), int(left_img.shape[0]/4)))
    disp_img_path = os.path.join(
        disp_img_root, img_name.split(".")[0] + '.npy')
    gt_disp = np.load(disp_img_path)

    _, img_w, _ = left_img.shape
    downsample_intrinsic = Zed_w/img_w
    fx = Zed_fx/downsample_intrinsic
    cx = Zed_cx/downsample_intrinsic
    cy = Zed_cy/downsample_intrinsic
    baseline = Zed_baseline

    depth_img = fx * baseline / (gt_disp + 1e-8)
    depth_img[gt_disp <= 0] = 0
    if label in label_nums:
        filename = label + "_" + str(label_nums[label])
        label_nums[label] += 1
    else:
        label_nums[label] = 2
        filename = label + "_1" 
    s_path = os.path.join(s_plt_path, filename + '.ply')
    img_path = os.path.join(s_plt_path, filename + '.png')
    cv2.imwrite(img_path, mask)
    save_plt(s_path, depth_img, mask, color, fx, cx, cy)

def process_depth(img_name, mask, label, color, s_plt_path):
    left_img_path = os.path.join(left_img_root, img_name+'.png')
    left_img = cv2.imread(left_img_path)
    left_img = cv2.resize(
        left_img, (int(left_img.shape[1]/4), int(left_img.shape[0]/4)))
    disp_img_path = os.path.join(
        disp_img_root, img_name.split(".")[0] + '.npy')
    gt_disp = np.load(disp_img_path)

    _, img_w, _ = left_img.shape
    downsample_intrinsic = Zed_w/img_w
    fx = Zed_fx/downsample_intrinsic
    cx = Zed_cx/downsample_intrinsic
    cy = Zed_cy/downsample_intrinsic
    baseline = Zed_baseline
    depth_img = fx * baseline / (gt_disp + 1e-8)
    depth_img[gt_disp <= 0] = 0
    # plt.imshow(depth_img)
    # plt.show()
    depth_img[mask != 255] = 0
    if label in['tree', 'pole','pebble', 'pillar']:
        mean_depth = get_mean_depth(depth_img)
        depth_img[abs(depth_img-mean_depth) > 2000.] = 0
    depth_img[depth_img > (30 * 1000.)] = 0

    # plt.imshow(label_polygon)
    # plt.show()
    return depth_img

    # save_plt(depth_img, mask, color, fx, cx, cy)

def clearlabelnums():
    label_nums.clear()