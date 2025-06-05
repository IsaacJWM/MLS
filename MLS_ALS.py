import laspy
import matplotlib.pyplot as plt
import geopandas as gpd
import open3d as o3d
import numpy as np
import copy
import os
import re
import time

MLS_plots = [
    '3020', '3021'
]

ALS_plots = [
    [625500, 1011500], [625500, 1012000],
    [626000, 1011500], [626000, 1012000],
    [626500, 1011500], [626500, 1012000],
]

Bounding_target = np.array([[0.0,0.0,0.0], [0.0,20.0,0.0], [20.0,0.0,0.0], [20.0,20.0,0.0]])

def calculate_target_xy(plot_num):
    target_q20 = CHP[CHP['P20'] == f'{plot_num[0]}-{plot_num[1]}']
    geom = target_q20.iloc[0].geometry
    rotated = geom.minimum_rotated_rectangle
    corners_unordered = np.array(rotated.exterior.coords)[:-1]
    corners_ordered = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])
    for corner in corners_unordered:
        if corner[0] == min(corners_unordered[:,0]):
            corners_ordered[0] = corner
        elif corner[1] == max(corners_unordered[:,1]):
            corners_ordered[1] = corner
        elif corner[1] == min(corners_unordered[:,1]):
            corners_ordered[2] = corner
        elif corner[0] == max(corners_unordered[:,0]):
            corners_ordered[3] = corner
    return corners_ordered

def lazO3d(file_path):
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def compute_transformation(source, target):
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute optimal rotation using SVD
    U, _, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R = U @ Vt  # Rotation matrix

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute translation
    t = target_mean - R @ source_mean

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation


def get_b_box(traj, z_extent):
    b_box = traj.get_minimal_oriented_bounding_box()
    extent = b_box.extent.copy()
    extent[0] += 10
    extent[1] += 10
    extent[2] = z_extent
    center = b_box.center.copy()
    center[2] = source_traj.get_min_bound()[2] + extent[2] * 0.5
    b_box = o3d.geometry.OrientedBoundingBox(center, b_box.R, extent)
    return b_box


def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result


ALSpath = r"C:\Users\iwrig\OneDrive\Desktop\2025_Summer\MLS\ALS data"
MLSpath = r"C:\Users\iwrig\OneDrive\Desktop\2025_Summer\MLS\Test"
CHPpath = r"C:\Users\iwrig\OneDrive\Desktop\2025_Summer\MLS\shapebci50ha\shapebci50ha\bci_20x20.shp"
path = os.path.join(ALSpath, 'BCI_2023_' + str(ALS_plots[3][0]) + '_' + str(ALS_plots[3][1]) + '.laz')
ALS_pcd = lazO3d(path)
ALS_points = np.asarray(ALS_pcd.points)
CHP = gpd.read_file(CHPpath) #Object will have a column, accessible as a pandas frame, that says q20.

threshold = 1
score_threshold = 0.99

for plot in MLS_plots:
    output_path_cloud = os.path.join(MLSpath, f"{plot}_cloud_transformed.ply")
    if True:
        target_position_xy = calculate_target_xy([int(plot[:2]),int(plot[2:])])
        target_position = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
        counter = 0
        #NOTE: To do this for the entire 50 ha plot, we would need a method of determining which ALS plot each MLS plot is in.
        for corner in target_position_xy:
            bounds = np.array([[corner[0] - 0.5, corner[0] + 0.5], [corner[1] - 0.5, corner[1] + 0.5]])
            xy = ALS_points[:, :2]
            mask = np.all((xy >= bounds[:, 0]) & (xy <= bounds[:, 1]), axis=1)
            z_vals = ALS_points[mask, 2]
            if z_vals.size > 0:
                target_position[counter] = [corner[0], corner[1], np.min(z_vals)]
            else:
                print("No points in the specified segment.")
            counter += 1

        #Reading files
        source_ref = np.loadtxt(os.path.join(MLSpath, rf'{plot}results_trajref.txt'), skiprows=1)
        source_traj = o3d.io.read_point_cloud(os.path.join(MLSpath, rf'{plot}results_traj_time.ply'))
        source_pcd_uncropped = lazO3d(os.path.join(MLSpath, rf'{plot}results.laz'))

        #Computing transformation
        transformation = compute_transformation(source_ref[:, :3][:4], target_position)
        R = transformation[:3, :3]  # Rotation
        t = transformation[:3, 3]  # Translation
        transformed_points = (R @ source_ref[:, :3][:4].T).T + t


        #Cropping point cloud
        z_diff = ALS_pcd.get_max_bound()[2] - ALS_pcd.get_min_bound()[2]
        b_box = get_b_box(source_traj, z_diff)
        source_pcd = source_pcd_uncropped.crop(b_box)

        #Transforming and writing source point cloud
        source_traj.transform(transformation)
        source_pcd.transform(transformation)
        o3d.io.write_point_cloud(output_path_cloud, source_pcd)
    else:
        source_pcd = lazO3d(output_path_cloud)
    transformation = np.eye(4)
    #_, _, _, result_matrix = icp(source_pcd, ALS_pcd_cropped, threshold, transformation[0], 4, 100, 3.7)

    ALS_pcd_overlap = ALS_pcd.crop(get_b_box(source_traj, z_diff))
    fitness, _ = evaluate_registration(source_pcd, ALS_pcd_overlap, threshold, transformation)
    print(plot, fitness)
    fitness, _ = evaluate_registration(source_pcd, ALS_pcd_overlap, 0.1, transformation)
    print(plot, fitness)


    loss = o3d.pipelines.registration.TukeyLoss(k=3.7)
    p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        ALS_pcd_overlap,
        max_correspondence_distance=0.1,
        init=transformation,
        estimation_method=p2p_loss
    )
    print(icp_result)
    downsampled_source_pcd = source_pcd.voxel_down_sample(0.1)
    downsampled_ALS_pcd = ALS_pcd.voxel_down_sample(0.1)
    v_sizes = [0.3, 0.2, 0.1, 0]

    for size in v_sizes:
        t3 = time.time()
        ALS_pcd_overlap = ALS_pcd.crop(get_b_box(source_traj, z_diff))
        if size != 0:
            ALS_pcd_overlap.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=100))
            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                ALS_pcd_overlap,
                max_correspondence_distance=2 * size,
                init=transformation,
                estimation_method=p2p_loss
            )
        else:
            ALS_pcd_overlap.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                ALS_pcd_overlap,
                max_correspondence_distance=0.1,
                init=transformation,
                estimation_method=p2p_loss
            )
        source_pcd.transform(icp_result.transformation)
        source_traj.transform(icp_result.transformation)
        print(size)
        print(icp_result.fitness)
        t4 = time.time()
        print("Time to perform ICP:", t4 - t3)

    fitness, _ = evaluate_registration(source_pcd, ALS_pcd, threshold, transformation)
    print(plot, fitness)


