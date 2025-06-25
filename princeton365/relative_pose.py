import os
import argparse
import numpy as np
import subprocess
from princeton365.utils.utils_graph import load_graph, convert_multidigraph_to_digraph, extract_poses
from princeton365.utils.utils_io import get_undistorted_camera_matrix, serialize_intrinsics, read_rotation_from_json, rotation_matrix_from_angles_insta360
from princeton365.optimization.graph_optimization import pose_graph_optimization

def main():
    parser = argparse.ArgumentParser(description="Example script building paths for intrinsics, detected points, etc.")
    
    parser.add_argument("--gt_detected_points", required=True, help="Path to the gt_view detected points json file")
    parser.add_argument("--user_detected_points", required=True, help="Path to the user view detected points json file")
    parser.add_argument("--gt_intrinsics", required=True, help="Path to the gt view intrinsics file")
    parser.add_argument("--user_intrinsics", required=True, help="Path to the user view intrinsics file")
    parser.add_argument("--pose_graph", required=True, help="Path to the pose graph pickle file")
    parser.add_argument("--gt_trajectory", required=True, help="Path to the gt trajectory file")
    parser.add_argument("--resolution", default=(3840, 2160), type=tuple, help="Resolution of the videos")
    args = parser.parse_args()
    
    # Intrinsics
    gt_camera_matrix = np.load(os.path.join(args.gt_intrinsics, "camera_mtx.npy"))
    gt_dist_coeffs = np.load(os.path.join(args.gt_intrinsics, "camera_dist.npy"))
    gt_camera_matrix, _ = get_undistorted_camera_matrix(gt_camera_matrix, gt_dist_coeffs, args.resolution[0], args.resolution[1])
    gt_camera_matrix_str, gt_dist_coeffs_str = serialize_intrinsics(gt_camera_matrix, None)  

    user_camera_matrix = np.load(os.path.join(args.user_intrinsics, "camera_mtx.npy"))
    user_dist_coeffs = np.load(os.path.join(args.user_intrinsics, "camera_dist.npy"))
    user_camera_matrix, _ = get_undistorted_camera_matrix(user_camera_matrix, user_dist_coeffs, args.resolution[0], args.resolution[1])
    user_camera_matrix_str, user_dist_coeffs_str = serialize_intrinsics(user_camera_matrix, None)

    # Pose Graph
    G = load_graph(args.pose_graph)
    G = convert_multidigraph_to_digraph(G)
    G = pose_graph_optimization(G)

    board_data = extract_poses(G)
    board_data_str = ";".join(board_data)
    
    # Rotation given from the 360 camera. It will be used for the initialization to bundle_pnp
    base_path = os.path.dirname(args.user_intrinsics)
    rotation_360 = read_rotation_from_json(f'{base_path}/rotation_360.json')
    initial_guess = rotation_matrix_from_angles_insta360(rotation_360)
    initial_guess = initial_guess.flatten()
    initial_guess = ' '.join(str(x) for x in initial_guess)
    
    # Bundle Rig PnP
    cpp_program_path = "princeton365/optimization/rig_bundle_pnp/build/optimize_rig"
    command = [cpp_program_path, args.gt_detected_points, args.user_detected_points, gt_camera_matrix_str, gt_dist_coeffs_str, user_camera_matrix_str, user_dist_coeffs_str, board_data_str, args.gt_trajectory, initial_guess]
    try:       
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("C++ Program Output:")
        print(result.stdout) 
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ program: {e}")
        print(f"Program stderr: {e.stderr}")


if __name__ == "__main__":
    main()

