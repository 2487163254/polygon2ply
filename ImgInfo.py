import numpy as np




class ImgInfo:
    def __init__(self, png_dir, filename, polygons, labels, colors):
        self.png_dir = png_dir
        self.filename = filename
        self.polygons = polygons
        self.labels = labels
        self.colors = colors


    def __str__(self) -> str:
        return f"ImgInfo: {self.filename}; Dir:{self.png_dir}; Lens of polygons:{len(self.polygons)}; Labels:{self.labels}"


    def add_polygons(self, polygons):
        if polygons is None:
            return
        elif type(polygons) is list:
            for polygon in polygons:
                self.polygons.append(polygon)
        elif type(polygons) is np.array:
            self.polygons.append(polygons)
        else:
            raise TypeError("Invalid polygons type")

    def add_labels(self, labels):
        if labels is None:
            return
        elif type(labels) is list:
            for label in labels:
                self.labels.append(label)
        elif type(labels) is str:
            self.labels.append(labels)
        else:
            raise TypeError("Invalid labels type")
    
    def add_color(self, colors):
        if colors is None:
            return
        elif type(colors) is list:
            for color in colors:
                self.colors.append(color)
        elif type(colors) is tuple:
            self.colors.append(colors)