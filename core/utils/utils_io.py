import os 
import cv2
import yaml
import json
import types
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def get_result_save_path( video_path, result_type="", filename_prefix="", extension="", output_path=""): 
    """
    Generate a path to save a result file based on video_path.

    Parameters:
    - video_path (str): Path to the original video.
    - result_type (str): Subdirectory name under results or output_path.
    - filename_prefix (str): Prefix for the saved file (e.g., "pose_graph_", "points_").
    - extension (str): File extension including dot (e.g., ".pickle", ".json").
    - output_path (str): If provided, this is used as a base and result_type + filename are added.

    Returns:
    - str: Full path to save the result file.
    """
    # Extract filename without extension
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    result_filename = f"{filename_prefix}{video_filename}{extension}"

    if output_path:
        full_path = os.path.join(output_path, result_type)
        os.makedirs(full_path, exist_ok=True)
        return os.path.join(full_path, result_filename)

    # Try to derive a sub_path based on 'Benchmark' or fallback to parent directory
    try:
        parts = video_path.split(os.sep)
        benchmark_idx = parts.index("Benchmark")
        sub_path_parts = parts[benchmark_idx + 1 : benchmark_idx + 4] # We care only about Benchmark/category/view
        sub_path = os.path.join(*sub_path_parts)
    except ValueError:
        sub_path = os.path.basename(os.path.dirname(video_path))

    full_path = os.path.join(os.getcwd(), "results", result_type, os.path.dirname(sub_path))
    os.makedirs(full_path, exist_ok=True)

    return os.path.join(full_path, result_filename)


def plot_flow_histogram(flows_per_frame, output_path):
    """
    Plot the distribution of induced optical flow as a histogram.
    
    Args:
        flows_per_frame: List of arrays containing flow magnitudes per frame
        output_path: Path to save the histogram plot
    """
    if not flows_per_frame or all(len(f) == 0 for f in flows_per_frame):
        print("No valid flow data to plot. Skipping histogram.")
        return
    
    # Concatenate flow magnitudes from all frames
    all_flows = np.concatenate(flows_per_frame)
    
    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.hist(all_flows, bins=100, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('Optical Flow Magnitude (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Induced Optical Flow')
    plt.grid(alpha=0.3)
    
    # Add statistics to the plot
    mean_flow = np.mean(all_flows)
    median_flow = np.median(all_flows)
    max_flow = np.max(all_flows)
    plt.axvline(mean_flow, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_flow:.2f}')
    plt.axvline(median_flow, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_flow:.2f}')
    plt.legend()
    
    plt.savefig(output_path)
    print(f"Flow histogram saved to {output_path}")

def numpy_to_list(obj):
    """
    Recursively converts NumPy arrays in the object to Python lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    else:
        return obj


def save_board_points_to_json(board_points, output_path):
    """
    Converts a board_points dictionary into a C++-friendly JSON format and writes it to a file.
    Automatically handles NumPy arrays by converting them to regular Python lists.
    
    Parameters:
    - board_points (dict): Mapping of frame indices to board detections.
    - output_path (str): Path where the resulting JSON file will be saved.
    """
    json_data = []

    for frame_index, boards_in_frame in board_points.items():
        for _, boards in boards_in_frame:
            for board in boards:
                board_id = board[0]
                board_3d = numpy_to_list(board[1])
                board_2d = numpy_to_list(board[2].reshape(-1, 2))

                json_data.append({
                    "f": frame_index,
                    "b_id": board_id,
                    "3d": board_3d,
                    "2d": board_2d
                })

    with open(output_path, "w") as f:
        json.dump(json_data, f)



def read_rotation_from_json(filepath):
    '''
    Reads the rotation angles from a JSON file.
    The JSON file should contain a dictionary with keys "pan", "tilt", and "roll",
    each associated with a float value representing the angle in degrees.
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
        pan = np.deg2rad(data["rotation"]["pan"])
        tilt = np.deg2rad(data["rotation"]["tilt"])
        roll = np.deg2rad(data["rotation"]["roll"])
    return pan, tilt, roll


def rotation_matrix_from_angles_insta360(rotation):
    """
    Builds a 4x4 transform where 'tilt' is a rotation around X,
    'pan' is a rotation around Y, 'roll' is a rotation around Z,
    in the order: roll -> pan -> tilt.
    
    According to your experiments:
      - X-axis = tilt
      - Y-axis = pan
      - Z-axis = roll
      - The working order is zyx (roll, then pan, then tilt).
    
    Parameters
    ----------
    rotation : dict
        A dictionary with keys 'pan', 'tilt', 'roll' in degrees, e.g.:
            {
                "pan":  10.0,
                "tilt": 15.0,
                "roll": 20.0
            }

    Returns
    -------
    T : np.ndarray of shape (4, 4)
        A homogeneous transform matrix. The upper-left 3x3 is the
        final rotation. 
    """
    pan, tilt, roll = rotation
    
    # So we pass [roll_deg, pan_deg, tilt_deg] in that order.
    rot_mat_3x3 = R.from_euler('zyx', [roll, pan, tilt]).as_matrix()
    
    # Build a 4x4 homogeneous transform
    T = np.eye(4)
    T[:3, :3] = rot_mat_3x3
    return T

def rotation_distance(R1, R2):
    """Geodesic distance between rotations"""
    R_diff = R1 @ R2.T
    theta = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    return theta



def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray, with_scaling = False) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    if with_scaling:
        s = np.trace(np.diag(D) @ S) / sx
        t = my - s * (R @ mx)
    else:
        t = my - (R @ mx)
        s = 1.0


    return R, s, t



def get_undistorted_camera_matrix(camera_matrix, dist_coeffs, width, height):
    """
    Computes the optimal camera matrix for undistorting images.
    
    Parameters:
    - camera_matrix (np.ndarray): Original camera matrix.
    - dist_coeffs (np.ndarray): Distortion coefficients.
    - width (int): Width of the image.
    - height (int): Height of the image.

    Returns:
    - np.ndarray: Optimal camera matrix for undistortion.
    - tuple: Region of interest (ROI) for the undistorted image.
    """
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    return new_camera_matrix, roi 


def serialize_intrinsics(camera_matrix, dist_coeffs ):
    '''
    Serializes camera intrinsics and distortion coefficients into a string format.
    '''
    camera_matrix_str = ",".join(map(str, camera_matrix.flatten()))
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    dist_coeffs_str = ",".join(map(str, dist_coeffs.flatten()))
    return camera_matrix_str, dist_coeffs_str


def load_config_from_yaml(board_type="charuco", path="configs/board_configs.yaml"):
    '''
    Loads configuration from a YAML file.
    '''
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)

    if board_type not in yaml_data:
        raise ValueError(f"'{board_type}' config not found in {path}")

    return types.SimpleNamespace(**yaml_data[board_type])