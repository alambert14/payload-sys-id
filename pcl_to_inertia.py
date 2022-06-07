import numpy as np
import open3d as o3d

from manipulation.meshcat_utils import draw_open3d_point_cloud, draw_points
from manipulation.open3d_utils import create_open3d_point_cloud


def pcl_to_voxel(pcl: o3d.geometry.PointCloud, voxel_size: float = 0.001):
    """
    Convert a pointcloud from drake into a voxel grid

    :param pcl: Pointcloud from drake camera
    :param voxel_size: size of the voxel for each point in the pointcloud
    :return: A voxel grid, where each voxel corresponds to a pointcloud point
    """
    # o3d_pcl = create_open3d_point_cloud(pcl)
    # o3d.visualization.draw_geometries([pcl])
    voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxels])
    return voxels


def calculate_moment_of_inertia_origin(voxel_grid: o3d.geometry.VoxelGrid,
                                       total_mass: float, voxel_size = 0.001):
    voxels = voxel_grid.get_voxels()
    voxel_mass = total_mass / len(voxels)
    print(len(voxels))

    # TODO: Find the center of mass, and then find the moment of inertia wrt the com
    coords = np.array([voxel.grid_index for voxel in voxels]) * voxel_size
    print(coords.shape)
    com = np.sum(coords, axis=0) / len(voxels)  # * voxel_mass / total_mass
    print(com)

    # TODO: vectorize this
    Ixx = 0
    Iyy = 0
    Izz = 0
    Ixy = 0
    Iyz = 0
    Ixz = 0

    Gxx = 0
    Gyy = 0
    Gzz = 0
    Gxy = 0
    Gyz = 0
    Gxz = 0
    for voxel in voxels:
        coord = (voxel.grid_index * voxel_size)
        Gxx += coord[0] ** 2
        Gyy += coord[1] ** 2
        Gzz += coord[2] ** 2
        Gxy += coord[0] * coord[1]
        Gxz += coord[0] * coord[2]
        Gyz += coord[1] * coord[2]
        # Ixx += voxel_mass * (coord[1] ** 2 + coord[2] ** 2)
        # Iyy += voxel_mass * (coord[0] ** 2 + coord[2] ** 2)
        # Izz += voxel_mass * (coord[0] ** 2 + coord[1] ** 2)
        # Ixy -= voxel_mass * coord[0] * coord[1]
        # Iyz -= voxel_mass * coord[1] * coord[2]
        # Ixz -= voxel_mass * coord[0] * coord[2]

    Gxx /= len(voxels)
    Gyy /= len(voxels)
    Gzz /= len(voxels)
    Gxy /= len(voxels)
    Gyz /= len(voxels)
    Gxz /= len(voxels)

    Gxx *= total_mass
    Gyy *= total_mass
    Gzz *= total_mass
    Gxy *= total_mass
    Gyz *= total_mass
    Gxz *= total_mass

    print(np.array([total_mass, com[0], com[1], com[2], Gxx, Gyy, Gzz, Gxy, Gxz, Gyz]))
    return np.array([total_mass, com[0], com[1], com[2], Ixx, Iyy, Izz, Ixy, Ixz, Iyz])

def calculate_interia_tensor_from_voxels(voxel_grid: o3d.geometry.VoxelGrid,
                                         total_mass: float, voxel_size = 0.001):
    """
    Find the inertia tensor of a voxelized object given a per-unit mass
    :param voxel_grid: Voxelized object
    :param voxel_mass: Mass of each voxel
    :return: np.array - intertia tensor
    """
    voxels = voxel_grid.get_voxels()
    voxel_mass = total_mass / len(voxels)
    print(len(voxels))

    # TODO: Find the center of mass, and then find the moment of inertia wrt the com
    coords = np.array([voxel.grid_index for voxel in voxels]) * voxel_size
    print(coords.shape)
    com = np.sum(coords, axis=0) / len(voxels)  #  * voxel_mass / total_mass
    print(com)

    # TODO: vectorize this
    # Gxx = 0
    # Gyy = 0
    # Gzz = 0
    # Gxy = 0
    # Gyz = 0
    # Gxz = 0
    Ixx = 0
    Iyy = 0
    Izz = 0
    Ixy = 0
    Iyz = 0
    Ixz = 0
    for voxel in voxels:
        coord = (voxel.grid_index * voxel_size) - com
        # coord = (voxel.grid_index * voxel_size)
        # Gxx += coord[0] ** 2
        # Gyy += coord[1] ** 2
        # Gzz += coord[2] ** 2
        # Gxy += coord[0] * coord[1]
        # Gxz += coord[0] * coord[2]
        # Gyz += coord[1] * coord[2]
        Ixx += voxel_mass * (coord[1] ** 2 + coord[2] ** 2)
        Iyy += voxel_mass * (coord[0] ** 2 + coord[2] ** 2)
        Izz += voxel_mass * (coord[0] ** 2 + coord[1] ** 2)
        Ixy -= voxel_mass * coord[0] * coord[1]
        Iyz -= voxel_mass * coord[1] * coord[2]
        Ixz -= voxel_mass * coord[0] * coord[2]
    # Gxx /= len(voxels)
    # Gyy /= len(voxels)
    # Gzz /= len(voxels)
    # Gxy /= len(voxels)
    # Gyz /= len(voxels)
    # Gxz /= len(voxels)
    #
    # Gxx *= total_mass
    # Gyy *= total_mass
    # Gzz *= total_mass
    # Gxy *= total_mass
    # Gyz *= total_mass
    # Gxz *= total_mass

    print(np.array([total_mass, com[0], com[1], com[2], Ixx, Iyy, Izz, Ixy, Ixz, Iyz]))
    return np.array([total_mass, com[0], com[1], com[2], Ixx, Iyy, Izz, Ixy, Ixz, Iyz])


def calculate_ground_truth_parameters(filename):
    pcl = o3d.io.read_point_cloud(filename)
    voxels = pcl_to_voxel(pcl)

    return calculate_interia_tensor_from_voxels(voxels, 0.603000)


if __name__ == '__main__':
    # Open the laserscan
    pcl = o3d.io.read_point_cloud('nontextured.ply')
    voxels = pcl_to_voxel(pcl)
    print(type(voxels))
    print(calculate_interia_tensor_from_voxels(voxels, 0.603000))
