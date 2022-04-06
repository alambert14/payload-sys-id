import numpy as np
import open3d as o3d

from manipulation.meshcat_utils import draw_open3d_point_cloud, draw_points
from manipulation.open3d_utils import create_open3d_point_cloud


def pcl_to_voxel(pcl: o3d.geometry.PointCloud, voxel_size: float = 0.005):
    """
    Convert a pointcloud from drake into a voxel grid

    :param pcl: Pointcloud from drake camera
    :param voxel_size: size of the voxel for each point in the pointcloud
    :return: A voxel grid, where each voxel corresponds to a pointcloud point
    """
    o3d_pcl = create_open3d_point_cloud(pcl)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcl, voxel_size=voxel_size)


def calculate_interia_tensor_from_voxels(voxel_grid: o3d.geometry.VoxelGrid,
                                         voxel_mass: float = 0.01):
    """
    Find the inertia tensor of a voxelized object given a per-unit mass
    :param voxel_grid: Voxelized object
    :param voxel_mass: Mass of each voxel
    :return: np.array - intertia tensor
    """
    voxels = voxel_grid.get_voxels()
    # TODO: vectorize this
    Ixx = 0
    Iyy = 0
    Izz = 0
    Ixy = 0
    Iyz = 0
    Ixz = 0
    for voxel in voxels:
        coord = voxel.grid_index()
        Ixx += voxel_mass * (coord[1] ** 2 + coord[2] ** 2)
        Iyy += voxel_mass * (coord[0] ** 2 + coord[2] ** 2)
        Izz += voxel_mass * (coord[0] ** 2 + coord[1] ** 2)
        Ixy -= voxel_mass * coord[0] * coord[1]
        Iyz -= voxel_mass * coord[1] * coord[2]
        Ixz -= voxel_mass * coord[0] * coord[2]

    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixx, Iyz, Izz]])
