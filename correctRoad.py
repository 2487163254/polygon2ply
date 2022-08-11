import open3d as o3d
import os

ROOT = os.getcwd()
plys_dir = os.path.join(ROOT, 'export')
filenames = os.listdir(plys_dir)


def getplane(dir):
    pcd = o3d.io.read_point_cloud(dir)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                            ransac_n=200,
                                            num_iterations=3000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])



dir = os.path.join(plys_dir, filenames[0])
getplane(dir)