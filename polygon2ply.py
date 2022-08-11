import cv2
import numpy as np
from utils import get_ply, clearlabelnums, process_depth, presave_ply
import os
import matplotlib.pyplot as plt

def polygon2ply(img_name: str, polygons: list, labels: list, colors: tuple, export_dir: str) -> None:
    """
    img_name 是不带文件后缀
    polygons 按照fillPoly的要求是 list[np.array[list[list[int]]]]
    labels 标签
    color 0-255的tuple 3个数
    export_dir 文件夹

    disp img 这两个文件夹不要改;)
    需要确保polygons, labels, colors 三个list长度相同
    """

    if len(polygons) != len(labels) or len(polygons) != len(colors):
        raise ValueError(
            "The length of the input Polygons and Labels and Colors is not equal.")
    ply_dir = os.path.join(export_dir, img_name)
    for i in range(len(polygons)):
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygons[i]).astype(np.int32)], color=255)
        mask = cv2.resize(mask, (int(1920/4), int(1080/4)))

        # check folder
        if os.path.exists(ply_dir):
            pass
        else:
            os.mkdir(ply_dir)
        # get_ply(img_name, mask, labels[i], colors[i], ply_dir)
        process_depth(img_name, mask, labels[i], colors[i], ply_dir)
    clearlabelnums()


def filter_depth(img_name: str, polygons: list, labels: list, colors: tuple, export_dir: str):
    if len(polygons) != len(labels) or len(polygons) != len(colors):
        raise ValueError(
            "The length of the input Polygons and Labels and Colors is not equal.")
    ply_dir = os.path.join(export_dir, img_name)
    img = cv2.imread(os.path.join(os.getcwd(),'img', img_name+'.png'))
    img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
    for i in range(len(polygons)):
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygons[i]).astype(np.int32)], color=255)
        mask = cv2.resize(mask, (int(1920/4), int(1080/4)))

        # check folder
        if os.path.exists(ply_dir):
            pass
        else:
            os.mkdir(ply_dir)
        # get_ply(img_name, mask, labels[i], colors[i], ply_dir)
        depth_label = process_depth(img_name, mask, labels[i], colors[i], ply_dir)
        
        img[depth_label != 0] = (colors[i][2], colors[i][1], colors[i][0])
    cv2.imwrite(os.path.join(os.getcwd(), 'labeled', img_name + '.png'), img)



def getroadply(img_name: str, polygons: list, labels: list, colors: tuple, export_dir: str) -> None:
    """
    img_name 是不带文件后缀
    polygons 按照fillPoly的要求是 list[np.array[list[list[int]]]]
    labels 标签
    color 0-255的tuple 3个数
    export_dir 文件夹

    disp img 这两个文件夹不要改;)
    需要确保polygons, labels, colors 三个list长度相同
    """

    if len(polygons) != len(labels) or len(polygons) != len(colors):
        raise ValueError(
            "The length of the input Polygons and Labels and Colors is not equal.")
    # ply_dir = os.path.join(export_dir, img_name)
    depth_label = np.zeros((270, 480))
    
    for i in range(len(polygons)):
        if labels[i] != 'road' and labels[i] != 'footway':
            continue
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygons[i]).astype(np.int32)], color=255)
        mask = cv2.resize(mask, (int(1920/4), int(1080/4)))


        depth_label += process_depth(img_name, mask, labels[i])
    presave_ply(depth_label, mask, img_name, colors[i], export_dir)
    # plt.imshow(depth_label)
    # plt.show()

    clearlabelnums()

# def main():
#     testpoly = [np.array([[1920, 1080], [0, 1080], [0, 621], [611, 557], [727, 554], [726, 595], [726, 621], [730, 643], [724, 645], [732, 652], [737, 657], [736, 674], [730, 687], [727, 693], [743, 707], [770, 705], [774, 687], [771, 657], [779, 649], [783, 645], [777, 618], [776, 604], [773, 601], [774, 582], [780, 574], [777, 560], [782, 552], [798, 552], [796, 574], [793, 608], [790, 626], [792, 643], [799, 651], [799, 667], [796, 690], [795, 708], [826, 708], [859, 704], [878, 699], [886, 695], [887, 690], [884, 685], [881, 679], [862, 674], [859, 670], [858, 664], [837, 661], [837, 654], [843, 649], [840, 643], [845, 636], [843, 629], [
#                  846, 602], [853, 577], [849, 570], [855, 567], [858, 546], [1036, 539], [1097, 538], [1097, 533], [1090, 526], [1089, 519], [1177, 519], [1180, 521], [1187, 519], [1190, 519], [1194, 520], [1199, 519], [1247, 519], [1275, 519], [1275, 521], [1284, 523], [1284, 520], [1291, 520], [1291, 523], [1300, 523], [1303, 526], [1309, 526], [1309, 523], [1313, 517], [1312, 514], [1313, 507], [1321, 507], [1324, 505], [1331, 507], [1335, 508], [1343, 510], [1379, 513], [1423, 511], [1425, 516], [1419, 524], [1435, 526], [1443, 530], [1545, 535], [1804, 545], [1808, 546], [1807, 548], [1794, 554], [1785, 570], [1789, 580], [1801, 591], [1920, 611]])]
#     img_name = '57e271b8-d6a4-11ec-b9e2-92640e831eb9'
#     export_dir = os.path.join(os.getcwd(), 'export')
#     polygon2ply(img_name, testpoly, (0, 255, 0), export_dir)


# if __name__ == '__main__':
#     main()
