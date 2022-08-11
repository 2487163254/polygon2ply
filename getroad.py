import os
from mergeJson import loadjson
from tqdm import tqdm
from polygon2ply import getroadply



def main():
    imgInfos = loadjson()
    for imgInfo in tqdm(imgInfos):
        getroadply(imgInfo.filename, imgInfo.polygons,
            imgInfo.labels, imgInfo.colors, "export")        


if __name__ == '__main__':
    main()