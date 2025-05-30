import numpy as np
import argparse
import os,tqdm
import open3d
from utils import walkDir

def load_kitti_bin(file_path, n_fts):
    """Load a KITTI .bin file into a NumPy array."""
    # print(f"Loading file from: {file_path}")
    if file_path.endswith(".bin"):
        return np.fromfile(file_path, dtype=np.float32).reshape(
            -1, n_fts
        )  # KITTI format: x, y, z, intensity
    elif file_path.endswith(".pcd"):
        return np.asarray(open3d.io.read_point_cloud(file_path).points)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")



def save_kitti_bin(file_path, point_cloud, n_fts):
    """Save a NumPy array as a KITTI .bin file."""
    # print(f"Saving file to: {file_path}")
    if file_path.endswith(".bin"):  
        point_cloud.reshape((-1, n_fts)).astype(np.float32).tofile(file_path)
    elif file_path.endswith(".pcd"):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud[:, :3])  # Only use x, y, z coordinates
        open3d.io.write_point_cloud(file_path, pcd)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def remove_half_beams(points):
    """
    Remove every alternate beam based on the z-values.
    Assuming the dataset has structured beams.
    """
    # Sort points by z-axis (height)
    sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[sorted_indices]

    # Select only alternate beams
    filtered_points = sorted_points[::2]

    return filtered_points


def process_kitti_bin(input_file, output_file, n_fts):
    """Process KITTI .bin file to remove half of the beams."""
    if file.endswith('pcd'):
        n_fts=3
    points = load_kitti_bin(input_file, n_fts)
    if (points.shape[0] == 0 or points.shape[1] != n_fts):
        print(f"File {input_file} has a zero size or the number of features is not {str(n_fts)}!!!")
        return
    filtered_points = remove_half_beams(points)
    if (filtered_points.shape[0] == 0 or filtered_points.shape[1] != n_fts):
        print(f"File {output_file} has a zero size or the number of features is not {str(n_fts)}!!!")
        return
    save_kitti_bin(output_file, filtered_points, n_fts)
    # print(f"Processed file saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove half the beams from a KITTI .bin point cloud."
    )
    parser.add_argument("input", help="Path to the input KITTI .bin file")
    parser.add_argument("n_fts", help="The number of features each point")

    args = parser.parse_args()
    
    for file in tqdm.tqdm(walkDir(args.input)):
        if file.endswith('bin') or file.endswith('pcd'):
            if not os.path.exists(file):
                print(f"Error: Input file {file} does not exist.")
            else:
                process_kitti_bin(file, file, int(args.n_fts))
