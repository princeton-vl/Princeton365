import os 
import cv2
import types
import pickle

import numpy as np
import matplotlib.pyplot as plt
import evo.tools.file_interface as file_interface

from evo.tools import plot as evo_plot
from scipy.spatial.transform import Rotation as R
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
import networkx as nx
from princeton365.utils.utils_io import estimate_similarity_transformation

def format_pose(frame_idx, world_T_camera):
    '''
    Formats the pose for the camera trajectory in TUM format.
    Parameters:
        frame_idx (int): The frame index.
        world_T_camera (numpy.ndarray): The transformation matrix from world to camera coordinates.
    Returns:
        str: The formatted pose string.
    '''
    translation = world_T_camera[:3, 3]
    rotation_matrix = world_T_camera[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return f"{frame_idx}.0 {translation[0]} {translation[1]} {translation[2]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"
 
def build_pose_graph_camera_trajectory_path(graph_path, camera_trajectory):
    '''
    Builds the camera trajectory path based on the graph path.
    '''
    # Step 1: Extract clean filename
    filename = os.path.basename(graph_path)
    filename = filename.removeprefix("pose_graph_").removesuffix(".pickle")

    # Step 2: Replace 'gt_view' with 'pose_graph_extra_frames'
    new_base_path = camera_trajectory.replace("gt_view", "pose_graph_extra_frames")
    new_dir = os.path.dirname(new_base_path)
    graph_camera_trajectory = os.path.join(new_dir, f"{filename}.txt")

    # Step 3: Get replacement — folder before "pose_graph_extra_frames" in graph_path
    graph_segments = graph_path.split(os.sep)
    try:
        idx = graph_segments.index("pose_graph_extra_frames")
        replacement = graph_segments[idx - 1]  # Folder before it
    except ValueError:
        raise ValueError("'pose_graph_extra_frames' not found in graph_path")

    # Step 4: Replace folder after "Benchmark" in graph_camera_trajectory
    trajectory_segments = graph_camera_trajectory.split(os.sep)
    try:
        bench_idx = trajectory_segments.index("camera_poses")
        trajectory_segments[bench_idx + 1] = replacement
    except ValueError:
        raise ValueError("'camera_poses' not found in camera trajectory path")

    # Rebuild the path
    graph_camera_trajectory = os.sep.join(trajectory_segments)

    return graph_camera_trajectory, replacement, filename


def parse_trajectory(camera_poses):
    """
    Parses a list of pose strings.
    
    Parameters:
        camera_poses (list of str): Each string should be formatted as:
            "timestamp x y z qx qy qz qw"
    Returns:
        PoseTrajectory3D: The parsed trajectory object.
    """
    timestamps = []
    positions = []
    orientations = []

    for pose_str in camera_poses:
        parts = pose_str.split()
        if len(parts) < 8:
            continue

        t = float(parts[0])
        x, y, z = map(float, parts[1:4])
        qw = float(parts[7])
        qx, qy, qz = map(float, parts[4:7])

        timestamps.append(t)
        positions.append([x, y, z])
        orientations.append([qw, qx, qy, qz])  # evo expects (w, x, y, z)

    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    orientations = np.array(orientations)

    # Create and return the trajectory object
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )

def save_trajectory_evo(traj, output_path):
    '''
    Saves the trajectory in TUM format.
    '''
    file_interface.write_tum_trajectory_file(output_path, traj)
    print(f"Trajectory saved in TUM format at {output_path}")


def plot_trajectory_evo(traj_gt, output_image_path, title="GT Trajectory", traj_est=None, ape=None, std=None):
    """
    Plots 2D (XY) ground truth trajectory and optionally an estimated trajectory with APE/STD.

    Parameters:
        traj_gt (PoseTrajectory3D): Ground truth trajectory.
        output_image_path (str): Path to save the output image.
        title (str): Plot title.
        traj_est (PoseTrajectory3D, optional): Estimated trajectory.
        ape (float, optional): Absolute Pose Error to display.
        std (float, optional): Standard deviation to display.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = evo_plot.prepare_axis(fig, evo_plot.PlotMode.xy)

    ax.set_title(title)

    evo_plot.traj(ax, evo_plot.PlotMode.xy, traj_gt, "-", "blue", "GT Trajectory")

    if traj_est is not None:
        evo_plot.traj(ax, evo_plot.PlotMode.xy, traj_est, "-", "red", "Estimated Trajectory")

    if ape is not None and std is not None:
        ax.text(0.95, 0.05,
                f"ATE: {ape:.6f} m\nStd Dev: {std:.6f} m",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))

    fig.savefig(output_image_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {output_image_path}")


def align_positions(traj_est, traj_gt):
    '''
    Aligns the estimated trajectory with the ground truth trajectory using similarity transformation.
    '''
    source = traj_est.positions_xyz.T  
    target = traj_gt.positions_xyz.T  
    
    R, s, t = estimate_similarity_transformation(source, target, True)
    aligned_positions = (s * (R @ source)).T + t.T

    return PoseTrajectory3D(
        positions_xyz=aligned_positions,
        orientations_quat_wxyz=traj_est.orientations_quat_wxyz,
        timestamps=traj_est.timestamps
    ), {"rotation": R, "scale": s, "translation": t}
    

def align_rotations(traj_est, traj_gt):
    '''
    Aligns the estimated trajectory with the ground truth trajectory using Kabsch algorithm.
    '''

    R_est_matrices = np.array([R.from_quat(np.roll(q, -1)).as_matrix() for q in traj_est.orientations_quat_wxyz])
    R_gt_matrices = np.array([R.from_quat(np.roll(q, -1)).as_matrix() for q in traj_gt.orientations_quat_wxyz])

    H = np.zeros((3, 3))
    for R_est, R_gt in zip(R_est_matrices, R_gt_matrices):
        H += R_est @ R_gt.T 

    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    R_opt = V @ U.T

    if np.linalg.det(R_opt) < 0:
        V[:, -1] *= -1
        R_opt = V @ U.T

    aligned_rotations = []
    for R_est in R_est_matrices:
        R_aligned = R_opt @ R_est
        quat_aligned = R.from_matrix(R_aligned).as_quat()
        # Convert from XYZW to WXYZ format
        quat_aligned_wxyz = np.roll(quat_aligned, 1)
        aligned_rotations.append(quat_aligned_wxyz)

    aligned_traj_est = PoseTrajectory3D(
        positions_xyz=traj_est.positions_xyz,
        orientations_quat_wxyz=np.array(aligned_rotations),
        timestamps=traj_est.timestamps
    )

    return aligned_traj_est, R_opt


def diagnose_trajectories(traj_est, traj_gt):
    """
    Diagnose potential issues with trajectories that might cause SVD alignment failure
    """
    print("\nTrajectory Diagnostics:")
    print("\n1. Basic Statistics:")
    print(f"Estimated trajectory shape: {traj_est.positions_xyz.shape}")
    print(f"Ground truth trajectory shape: {traj_gt.positions_xyz.shape}")
    print(f"Est NaN rows: {np.any(np.isnan(traj_est.positions_xyz), axis=1).sum()}")
    print(f"GT NaN rows: {np.any(np.isnan(traj_gt.positions_xyz), axis=1).sum()}")

    valid_est = ~np.any(np.isnan(traj_est.positions_xyz), axis=1)
    valid_gt = ~np.any(np.isnan(traj_gt.positions_xyz), axis=1)

    valid_both = valid_est & valid_gt
    valid_points_count = np.sum(valid_both)
    
    total_valid_gt = np.sum(valid_gt)
    valid_est_given_gt = np.sum(valid_gt & valid_est)

    print(f"\nValid ground truth frames: {total_valid_gt}")
    print(f"Valid estimated frames (where GT is valid): {valid_est_given_gt}")
    print(f"Valid corresponding points: {valid_points_count}")
    
    return valid_both, total_valid_gt, valid_est_given_gt


def pad_trajectory_with_nans(est_timestamps, est_positions_xyz, est_orientations_quat_wxyz, gt_timestamps):
    """
    Pad the estimated trajectory with NaN values where timestamps are missing compared to ground truth.
    
    Args:
        est_timestamps: Timestamps of the estimated trajectory
        est_positions_xyz: Positions of the estimated trajectory
        est_orientations_quat_wxyz: Orientations of the estimated trajectory in quaternion format
        gt_timestamps: Timestamps of the ground truth trajectory
        
    Returns:
        Padded timestamps, positions, and orientations for the estimated trajectory
    """

    est_pos_dict = {ts: pos for ts, pos in zip(est_timestamps, est_positions_xyz)}
    est_ori_dict = {ts: ori for ts, ori in zip(est_timestamps, est_orientations_quat_wxyz)}
    
    padded_positions = np.full((len(gt_timestamps), 3), np.nan)
    padded_orientations = np.full((len(gt_timestamps), 4), np.nan)
    
    for i, ts in enumerate(gt_timestamps):
        if ts in est_pos_dict:
            padded_positions[i] = est_pos_dict[ts]
            padded_orientations[i] = est_ori_dict[ts]
    
    padded_count = np.sum(np.isnan(padded_positions[:, 0]))
    
    return gt_timestamps, padded_positions, padded_orientations


def read_trajectory_file(filename, tum_format=True):

    timestamps, positions_xyz, orientations_quat_wxyz = [], [], []
    
    with open(filename, 'r') as file:
        for i, line in enumerate(file):         
            parts = line.strip().split()
            if not line or line.startswith('#'):
                continue   
            try:
                timestamp = float(parts[0])
                position = [float(parts[1]), float(parts[2]), float(parts[3])]

                # We need WXYZ format for the quaternions
                if tum_format:
                    orientation = [float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])]
                else:
                    orientation = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])] # w, x, y, z
                
                if (np.isnan(timestamp) or
                    any(np.isnan(position)) or
                    any(np.isnan(orientation))):
                    timestamps.append(timestamp)
                    positions_xyz.append([np.nan, np.nan, np.nan])
                    orientations_quat_wxyz.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    timestamps.append(timestamp)
                    positions_xyz.append(np.round(position, decimals=4))
                    orientations_quat_wxyz.append(np.round(orientation, decimals=4))

            except ValueError:
                continue

    return np.array(timestamps), np.array(positions_xyz), np.array(orientations_quat_wxyz)
 


# # This doesnt work
# def debug_frame():
#     text = f"Frame: {frame_idx} - ATE (m): {ate:.4f} - Final Pose Case: {final_pose_case}"
#     for i in range(count_valid):
#         text += f"\nBoard {valid_board_ids[i]} - Reprojection Error (p): {reproj_er[i]:.4f} - Points: {len(board_points['3d'][i])}"
#     for idx, board_instance in enumerate(valid_board_ids):
#         if board_instance is not None:
#             # Current frame's 3D points for this board
#             current_points = board_points["3d"][idx]                        
#             current_points_set = set(tuple(point) for point in current_points)
#             # Check if this board existed in the previous frame
#             if board_instance in hypothesis_2_board_points_previous:
#                 previous_points_set = set(
#                     tuple(point) for point in hypothesis_2_board_points_previous[board_instance]
#                 )

#                 # Calculate added and removed points
#                 added_points = current_points_set - previous_points_set
#                 removed_points = previous_points_set - current_points_set
#                 if added_points:
#                     text += f"\nBoard {board_instance}: {len(added_points)} p added."
#                     if removed_points:
#                         text += f"-{len(removed_points)} p removed."
#                 elif removed_points:
#                     text += f"\nBoard {board_instance}: {len(removed_points)} p removed."
#                 else:
#                     continue

#             else:
#                 # New board instance in the current frame
#                 text += f"\nBoard {board_instance}: {len(current_points_set)} points created. "
            
#             shortest_path = nx.shortest_path(
#                 G, source=valid_board_ids[0], target=board_instance, weight="distance"
#             )

#             T_local_to_board = np.eye(4)

#             for j in range(len(shortest_path) - 1):
#                 # edge_data = next(iter(G[shortest_path[j]][shortest_path[j + 1]].values()))
#                 edge_data = G[shortest_path[j]][shortest_path[j + 1]]
#                 T_local_to_board = np.dot(T_local_to_board, edge_data["transformation"])
#             translation_vector = T_local_to_board[:3, 3]
#             distance_to_first_board = np.linalg.norm(translation_vector)
#             text += f"-Distance : {distance_to_first_board:.4f} m."
    
#     for previous_board_instance in hypothesis_2_board_points_previous.keys():
#         if previous_board_instance not in valid_board_ids:
            
#             previous_points_set = set(
#                 tuple(point) for point in hypothesis_2_board_points_previous[previous_board_instance]
#             )
#             text += (
#                 f"\nBoard {previous_board_instance}: {len(previous_points_set)} p disappeared."
#             )
            
#             shortest_path = nx.shortest_path(
#                 G, source=valid_board_ids[0], target=previous_board_instance, weight="distance"
#             )

#             T_local_to_board = np.eye(4)

#             for j in range(len(shortest_path) - 1):
#                 # edge_data = next(iter(G[shortest_path[j]][shortest_path[j + 1]].values()))
#                 edge_data = G[shortest_path[j]][shortest_path[j + 1]]
#                 T_local_to_board = np.dot(T_local_to_board, edge_data["transformation"])
#             translation_vector = T_local_to_board[:3, 3]
#             distance_to_first_board = np.linalg.norm(translation_vector)
#             text += f"-Distance : {distance_to_first_board:.4f} m."
    
#     hypothesis_2_board_points_previous = {
#         valid_board_ids[idx]: board_points["3d"][idx]
#         for idx in range(len(valid_board_ids))
#         if valid_board_ids[idx] is not None
#     }
        
#     text += f"\nFinal Pose (m): {translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}"
#     text += f"\nFinal Quat: {quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_color = (77, 140, 32)
#     thickness = 3
#     line_height = 50
#     lines = text.split("\n")
#     padding = 10
#     background_color = (0, 0, 0)  
#     for i, line in enumerate(lines):
#         text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
#         text_width, text_height = text_size
#         x, y = 2500, 250 + i * line_height  # Adjust y-coordinate for each line

#         # Calculate rectangle coordinates
#         top_left = (x - padding, y - text_height - padding)
#         bottom_right = (x + text_width + padding, y + padding)

#         # Draw the background rectangle
#         cv2.rectangle(frame, top_left, bottom_right, background_color, -1)  # -1 fills the rectangle

#         # Draw the text over the rectangle
#         cv2.putText(
#             frame,
#             line,
#             (x, y),
#             font,
#             font_scale,
#             font_color,
#             thickness,
#         )
#     dot_images = cv2.resize(
#         dot_images, dim, interpolation=cv2.INTER_AREA
#     )
#     out_comparison.write(dot_images)



