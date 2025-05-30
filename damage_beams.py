import numpy as np
import argparse
import os, tqdm
import open3d
from utils_c import walkDir

def load_pcd(file_path, n_clmn):
    """Load a KITTI .bin file into a NumPy array."""
    # print(f"Loading file from: {file_path}")
    if file_path.endswith(".bin"):
        return np.fromfile(file_path, dtype=np.float32).reshape(
            -1, int(n_clmn)
        )  
    elif file_path.endswith(".pcd"):
        return np.asarray(open3d.io.read_point_cloud(file_path).points)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")



def save_pcd(file_path, point_cloud):
    """Save a NumPy array as a KITTI .bin file."""
    # print(f"Saving file to: {file_path}")
    if file_path.endswith(".bin"):  
        point_cloud.astype(np.float32).tofile(file_path)
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


def damage_beams(pcd, num_beams, dmg_beams_lst):
    max_z = np.max(pcd[:, 2])
    min_z = np.min(pcd[:, 2])
    span = (max_z-min_z)/float(num_beams)
    filtered_points = pcd
    for dmg_beam in dmg_beams_lst:
        bottom_z = min_z + span * dmg_beam
        top_z = min_z + span * (dmg_beam+1)
        filtered_points = filtered_points[~((filtered_points[:, 2] > bottom_z) & (filtered_points[:, 2] < top_z))]
    return filtered_points


def process_pcd(file, n_clmn, n_beams, lost_beam_lst):
    """Process KITTI .bin file to remove half of the beams."""
    points = load_pcd(file, n_clmn)
    if points.shape[0] == 0:
        return
    filtered_points = damage_beams(points, n_beams, lost_beam_lst)
    save_pcd(file, filtered_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove half the beams from a KITTI .bin point cloud."
    )
    parser.add_argument("path", help="Path to be processed")
    parser.add_argument("n_clmn", help="The number of columns each point")
    parser.add_argument("n_beams", help="The number of beams in point cloud")
    parser.add_argument("lost_beams", help="The lost beams list")

    args = parser.parse_args()
    
    lost_beam_lst = args.lost_beams.split(',')
    lost_beam_lst = [int(beam.strip()) for beam in lost_beam_lst]
    
    print('Begin to process point clouds with ', args.n_beams, ' beams...')
    print('Damaged beams include: ', end='')
    for beam in lost_beam_lst:
        print(str(beam), end=', ')
    print('\n\n\n')
    
    for file in tqdm.tqdm(walkDir(args.path)):
    # for file in walkDir(args.path):
        if file.endswith('bin') or file.endswith('pcd'):
            if not os.path.exists(file):
                print(f"Error: Input file {file} does not exist.")
            else:
                process_pcd(file, args.n_clmn, args.n_beams, lost_beam_lst)
