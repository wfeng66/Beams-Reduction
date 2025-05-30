import os, sys, random, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import open3d as o3d


def walkDir(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            abs_path = os.path.join(root, file)
            all_files.append(abs_path)
    return all_files


def load_ply(file_name, rt_arr=False):
    """
    Load a PLY file and return either an Open3D point cloud object or a numpy array of points.
    
    Parameters:
        file_name (str): The path to the PLY file.
        rt_arr (bool): If True, return a numpy array of points; otherwise, return an Open3D object.
    
    Returns:
        open3d.geometry.PointCloud or numpy.ndarray: The loaded point cloud as specified.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_name)
    
    if not pcd.has_points():
        print(f"The file '{file_name}' does not contain any valid points.")
        return None
    
    if rt_arr:
        # Convert Open3D point cloud to numpy array
        points = np.asarray(pcd.points)
        return points
    return pcd


def dist_filter(points, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
    """
    Filters the input array to retain points within the specified bounds using matrix operations.

    Parameters:
        points (numpy.ndarray): Array of shape (n, 3) containing the points.
        x_min, x_max, y_min, y_max, z_min, z_max (float or None): Bounds for filtering.
            If None, no filtering is applied along that axis.

    Returns:
        numpy.ndarray: Filtered array meeting the conditions.
    """
    # Create boolean masks for each axis based on the conditions
    mask_x = np.ones(points.shape[0], dtype=bool)
    mask_y = np.ones(points.shape[0], dtype=bool)
    mask_z = np.ones(points.shape[0], dtype=bool)

    if x_min is not None:
        mask_x = points[:, 0] >= x_min
    if x_max is not None:
        mask_x &= points[:, 0] <= x_max

    if y_min is not None:
        mask_y = points[:, 1] >= y_min
    if y_max is not None:
        mask_y &= points[:, 1] <= y_max

    if z_min is not None:
        mask_z = points[:, 2] >= z_min
    if z_max is not None:
        mask_z &= points[:, 2] <= z_max

    # Combine the masks for all axes
    combined_mask = mask_x & mask_y & mask_z

    # Return the filtered points
    return points[combined_mask]


def dist_filter1(points, x_min=-float("inf"), x_max=float("inf"), y_min=-float("inf"), y_max=float("inf"), z_min=-float("inf"), z_max=float("inf")):
    """
    Filters the input array to retain points within the specified bounds using matrix operations.

    Parameters:
        points (numpy.ndarray): Array of shape (n, 3) containing the points.
        x_min, x_max, y_min, y_max, z_min, z_max (float or None): Bounds for filtering.
            If None, no filtering is applied along that axis.

    Returns:
        numpy.ndarray: Filtered array meeting the conditions.
    """
    return points[(points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_min) & \
                  (points[:, 1] < y_max) & (points[:, 2] > z_min) & (points[:, 2] < z_max)]



def saveXYZ(abs_filename, point_cloud):
    np.savetxt(abs_filename, point_cloud, fmt="%.6f", delimiter=" ")


def save_ply(filepath, points):
    """
    Save point cloud data to a PLY file.

    Args:
        filepath (str): Path to save the PLY file.
        points (numpy.ndarray): Nx3 array containing point cloud data (X, Y, Z).
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    # Write the header and data to the file
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")





def visualize_ply(file_path):
    """
    Visualize a .ply file using Open3D.
    
    Args:
        file_path (str): Path to the .ply file to visualize.

    Example usage
    ply_file_path = "path/to/your/file.ply"
    visualize_ply(ply_file_path)
    """
    # Load the .ply file
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
        if not point_cloud.has_points():
            print(f"No points found in the .ply file: {file_path}")
            return
    except Exception as e:
        print(f"Error reading .ply file: {e}")
        return
    
    # Print basic information
    print(f"Loaded .ply file: {file_path}")
    print(f"Number of points: {len(point_cloud.points)}")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="PLY Point Cloud Visualization",
                                      width=800,
                                      height=600,
                                      point_show_normal=False,
                                      mesh_show_wireframe=False,
                                      mesh_show_back_face=False)


import numpy as np
import random
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import open3d as o3d

class PlaneEstimator(BaseEstimator, RegressorMixin):
    """
    Custom estimator for fitting a plane of the form ax + by + cz + d = 0.
    """
    def fit(self, X, y):
        p1, p2, p3 = X[:3]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        print(f"Sampled points: {p1, p2, p3}")
        print(f"Normal vector: {normal}")
        self.coef_ = normal[:2] / normal[2]  # Solve for a, b in terms of z = -(a*x + b*y + d)/c
        self.intercept_ = -np.dot(normal, p1) / normal[2]
        return self

    def predict(self, X):
        # Use the plane equation to compute z for given (x, y).
        return -(self.coef_[0] * X[:, 0] + self.coef_[1] * X[:, 1] + self.intercept_)

def compute_normals(points, k=10):
    """
    Compute normals for each point in the point cloud.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pc.normals)
    return normals

def validate_plane_normal(plane, ground_normal=np.array([0, 0, 1]), angle_threshold=10):
    """
    Validate the normal of the plane against the expected ground normal.
    """
    normal = np.array([*plane[:3]])
    normal /= np.linalg.norm(normal)
    angle = np.degrees(np.arccos(np.dot(normal, ground_normal)))
    return angle <= angle_threshold

def ransac_ground_plane(points, dist_threshold=0.05, max_z=-1.3, angle_threshold=10):
    """
    Extract ground plane using sklearn's RANSAC.
    """
    # Pre-process: Filter points with z < max_z
    ground_candidates = points[points[:, 2] < max_z]
    if len(ground_candidates) < 3:
        raise ValueError("Not enough points for RANSAC after filtering!")

    # Fit plane using RANSAC
    X = ground_candidates[:, :2]
    y = ground_candidates[:, 2]
    ransac = RANSACRegressor(
        estimator=PlaneEstimator(),
        residual_threshold=dist_threshold,
        min_samples=3,
        max_trials=5000,
    )
    ransac.fit(X, y)

    # Retrieve plane parameters
    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_
    plane = (a, b, c, d)

    # Validate plane
    if not validate_plane_normal(plane, angle_threshold=angle_threshold):
        raise ValueError("Fitted plane does not satisfy normal validation!")

    # Classify points based on the plane
    all_distances = np.abs(
        a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)
    ground_mask = all_distances < dist_threshold
    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]

    return plane, ground_points, non_ground_points


def viz_3dlist(files):
    for file in files:
        visualize_ply(file)
        
        
        
def iou(pred, target):
    """
    Calculates the Intersection over Union (IoU) for point cloud segmentation.

    Args:
        pred (torch.Tensor): Predicted segmentation labels (N, 1)
        target (torch.Tensor): Ground truth segmentation labels (N, 1)

    Returns:
        float: IoU score
    """

    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()

    if union == 0:
        return 1.0  # If no points are predicted or in the ground truth, IoU is 1

    iou_score = intersection.float() / union.float()
    return iou_score


def mean_iou(pred, target, num_classes):
    """
    Calculates the mean Intersection over Union (mIoU) for point cloud segmentation.

    Args:
        pred (torch.Tensor): Predicted segmentation labels (N, 1)
        target (torch.Tensor): Ground truth segmentation labels (N, 1)
        num_classes (int): Number of segmentation classes

    Returns:
        float: mIoU score
    """

    iou_per_class = []
    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        iou_per_class.append(iou(pred_class, target_class))

    return torch.mean(torch.stack(iou_per_class))


# def viz_3dlist(files):
#     """
#     Visualize a list of 3D point cloud file paths interactively.

#     Parameters:
#     - files (list of str): A list of file paths to 3D point cloud files.

#     Supported commands during runtime:
#     - Space/Enter: Visualize the next file.
#     - "b": Begin visualization from a specific file index.
#     - "q": Quit the visualization.
#     """
#     if not files:
#         print("The file list is empty.")
#         return

#     index = [0]  # Using a list to make it mutable in the callback scope

#     def update_visualization(vis):
#         print(f"Visualizing file at index {index[0]}: {files[index[0]]}")
#         vis.clear_geometries()
#         try:
#             pcd = load_ply(files[index[0]])
#             vis.add_geometry(pcd)
#         except Exception as e:
#             print(f"Error loading file {files[index[0]]}: {e}")
#         vis.poll_events()
#         vis.update_renderer()

#     def next_callback(vis):
#         index[0] = (index[0] + 1) % len(files)
#         update_visualization(vis)

#     def back_callback(vis):
#         try:
#             new_index = int(input(f"Enter new index (0 to {len(files) - 1}): ").strip())
#             if 0 <= new_index < len(files):
#                 index[0] = new_index
#                 update_visualization(vis)
#             else:
#                 print(f"Index out of range. Valid range is 0 to {len(files) - 1}.")
#         except ValueError:
#             print("Invalid input. Please enter an integer.")

#     def quit_callback(vis):
#         print("Exiting visualization.")
#         vis.close()

#     key_to_callback = {
#         ord(" "): next_callback,
#         ord("b"): back_callback,
#         ord("q"): quit_callback,
#     }

#     print(f"Visualizing file at index {index[0]}: {files[index[0]]}")
#     pcd = load_ply(files[index[0]])
#     o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)