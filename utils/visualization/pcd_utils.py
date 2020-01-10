
import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

import open3d as o3d

from datasets.base_patch_dataset import BasePointCloud

def pointcloud_to_o3d_pcd(cloud : BasePointCloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.pos.numpy())
    return pcd

def visualize_pointcloud(cloud):
    pcd = pointcloud_to_o3d_pcd(cloud)
    o3d.visualization.draw_geometries([pcd])