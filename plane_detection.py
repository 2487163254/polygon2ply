# import numpy as np
# from utils import PlaneRegression


# def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
#     plane_list = []
#     N = len(points)
#     target = points.copy()
#     count = 0

#     while count < (1 - min_ratio) * N:
#         w, index = PlaneRegression(
#             target, threshold=threshold, init_n=3, iter=iterations)
    
#         count += len(index)
#         plane_list.append((w, target[index]))
#         target = np.delete(target, index, axis=0)

#     return plane_list