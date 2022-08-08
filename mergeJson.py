import yaml
import json
import os
from ImgInfo import ImgInfo
from polygon2ply import polygon2ply, filter_depth
from tqdm import tqdm

'0f6780ec-b059-11ec-9d25-7c10c921acb3'
CLS_LIST = {"road": 1, "footway": 1, "pebble": 1, "grass": 1,
            "manhole": 1, "pillar": 2, "pole": 2, "pole": 2, "tree": 4}
COLOR_LIST = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203,
                                                192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51,
                                            153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128), ]
ROOT_DIR = os.getcwd()


with open('allnames.yaml', 'r') as f:
    ALL_NAMES = yaml.load(f, Loader=yaml.Loader)


def loadjson():
    json_labelme_dir = os.path.join(ROOT_DIR, 'json_labelme')
    morejson_dir = os.path.join(ROOT_DIR, 'morejson')
    img_root_dir = os.path.join(ROOT_DIR, 'img')
    ImgInfos = []
    for name in ALL_NAMES:

        # parse json file in json_labelme
        json_dir = os.path.join(json_labelme_dir, name + '.json')

        with open(json_dir, 'r') as f:
            data = json.load(f)
        png_dir = os.path.join(img_root_dir, name + '.png')
        polygons = []
        labels = []
        colors = []

        for shape in data['shapes']:
            # continue
            try:
                label = shape['label']
            except:
                raise Exception('Could not parse label from shape: %s, file: %s' % (
                    shape['label'], json_dir))
            if label in CLS_LIST:
                labels.append(label)
                polygons.append(shape['points'])
                colors.append(COLOR_LIST[CLS_LIST[label]])
        # parse json file in morejson

        json_dir = os.path.join(morejson_dir, name + '.json')
        with open(json_dir, 'r') as f:
            data = json.load(f)

        for object in data['objects']:
            try:
                label = object['label']
            except:
                raise Exception('Could not parse label from shape: %s, file: %s' % (
                    shape['label'], json_dir))
            if label in CLS_LIST:
                labels.append(label)
                polygons.append(object['polygon'])
                colors.append(COLOR_LIST[CLS_LIST[label]])

        imginfo = ImgInfo(png_dir, name, polygons, labels, colors)
        ImgInfos.append(imginfo)

    return ImgInfos


def main():
    ImgInfos = loadjson()
    for imgInfo in tqdm(ImgInfos):
        # if imgInfo.filename == '0f6780ec-b059-11ec-9d25-7c10c921acb3':
        filter_depth(imgInfo.filename, imgInfo.polygons,
                    imgInfo.labels, imgInfo.colors, "export")


if __name__ == '__main__':
    main()
