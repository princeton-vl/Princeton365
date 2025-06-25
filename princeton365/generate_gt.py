import os
import cv2
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from princeton365.utils.utils_io import load_config_from_yaml, get_result_save_path, serialize_intrinsics, get_undistorted_camera_matrix, save_board_points_to_json
from princeton365.utils.utils_aruco import detect_pose, multiboard_pnp, estimate_relative_pose, undistort_frame
from princeton365.utils.utils_graph import convert_multidigraph_to_digraph, median_edge_selection, get_valid_boards, get_root_board_index, save_graph, plot_graph, load_graph, snap_coordinate_systems, extract_poses
from princeton365.utils.utils_trajectory import parse_trajectory, save_trajectory_evo, plot_trajectory_evo, format_pose, build_pose_graph_camera_trajectory_path
from collections import deque, defaultdict
from princeton365.board_generator import generate_charuco_boards  
from princeton365.optimization.graph_optimization import pose_graph_optimization
import evo.tools.file_interface as file_interface
import networkx as nx

class Princeton365:
    def __init__(self, args):
        self.video = args.video
        self.camera_matrix = np.load(os.path.join(args.intrinsics, "camera_mtx.npy"))
        self.dist_coeffs = np.load(os.path.join(args.intrinsics, "camera_dist.npy"))
        self.output_path = args.output_path
        
        # Options
        self.use_pose_graph = args.use_pose_graph
        self.snap = args.snap
        self.skip_camera_poses = args.skip_camera_poses
        self.bundle_pnp = args.bundle_pnp
       
        # Board generation
        self.board_type = args.board_type
        self.config = args.config
        self.dictionary = None
        self.boards = self.generate_boards()

        # Debugging
        self.debug = args.debug


    def generate_boards(self):
        board_args = load_config_from_yaml(self.board_type, self.config)
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_args.dictionary))
        return generate_charuco_boards(board_args)
    
    def add_empty_pose(self, camera_poses, frame_idx, per_frame_board_points):
        pose_str = f"{frame_idx}.0 NaN NaN NaN NaN NaN NaN NaN"
        camera_poses.append(pose_str)
        if frame_idx not in per_frame_board_points:
            per_frame_board_points[frame_idx] = []
        per_frame_board_points[frame_idx].append((0,[]))
        return camera_poses, per_frame_board_points

    def get_video_info_and_camera_matrix(self):
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            raise ValueError(f"Error opening video stream or file: {self.video}")
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_camera_mtx, roi = get_undistorted_camera_matrix(self.camera_matrix, self.dist_coeffs, width, height)

        return total_frames, new_camera_mtx, roi, cap

    def generate_pose_graph(self, window_size=1000):
        '''
        Generate the pose graph of the video.
        '''
        total_frames, new_camera_mtx, roi, cap = self.get_video_info_and_camera_matrix()
        
        G = nx.MultiDiGraph()
        pair_detected = False
        pair_buffer = deque(maxlen=window_size)
        pair_freq = defaultdict(int)
        single_board_ids = set()
        
        for frame_idx in tqdm(range(total_frames), desc="Building Pose Graph"):
            ret, dist_frame = cap.read()
            if not ret:
                raise ValueError(f"Error reading frame {frame_idx} from video: {self.video}")
            
            frame = undistort_frame(dist_frame, self.camera_matrix, self.dist_coeffs, new_camera_mtx, roi)       
            
            # Pose estimation between Boards
            relative_poses, pair_detected, single_board = estimate_relative_pose(
                frame, new_camera_mtx, None, self.boards, self.dictionary, pair_detected
            )
            # Add detected pairs to Graph
            if pair_detected:
                for detected_boards, T in relative_poses:
                    key = tuple(sorted(detected_boards))
                    if pair_freq[key] >= window_size:
                        continue

                    if key in pair_buffer:
                        pair_freq[key] += 1
                    else:
                        pair_buffer.clear()
                        pair_freq.clear()
                        pair_buffer.append(key)
                        pair_freq[key] = 1

                    board1, board2 = detected_boards
                    board1_node = f"{board1}_0"
                    board2_node = f"{board2}_0"

                    if board1_node not in G.nodes:
                        if board2_node not in G.nodes:
                            G.add_node(board1_node, pose=np.eye(4))
                        else:
                            G.add_node(
                                board1_node,
                                pose=np.dot(G.nodes[board2_node]["pose"], T),
                            )

                    if board2_node not in G.nodes:
                        if board1_node in G.nodes:
                            G.add_node(
                                board2_node,
                                pose=np.dot(G.nodes[board1_node]["pose"], T),
                            )
                        else:
                            print(f"Board {board1_node} not found in the graph.")

                    G.add_edge(
                        board1_node,
                        board2_node,
                        transformation=T,
                        distance=round(np.linalg.norm(T[:3, 3]), 3),
                    )
                    G.add_edge(
                        board2_node,
                        board1_node,
                        transformation=np.linalg.inv(T),
                        distance=round(np.linalg.norm(T[:3, 3]), 3),
                    )
                    pair_buffer.append(key)
            else:
                single_board_ids.add(single_board)
        
        cap.release()
        
        pose_graph_path_pkl = get_result_save_path(self.video, result_type="pose_graphs", filename_prefix = "pose_graph_", extension = ".pickle", output_path = self.output_path)
        plot_path = pose_graph_path_pkl.replace(".pickle", ".png")
       
        if not pair_detected:
            single_board_ids = {b for b in single_board_ids if b is not None}
            if len(single_board_ids) > 1:
                raise ValueError("The set contains more than 2 items")
            single_board = next(iter(single_board_ids))
            G.add_node(f"{single_board}_0", pose=np.eye(4))
            if self.debug:
                plot_graph(G, title="Pose Graph", save_path=plot_path)
        else:
            G = median_edge_selection(G)
            edge_labels = {
                (u, v): f"{d['distance']} m" 
                for u, v, d in G.edges(data=True)
            }
            if self.debug: 
                plot_graph(G, title="Pose Graph", edge_labels=edge_labels, save_path=plot_path)
        save_graph(G, pose_graph_path_pkl)
        return G


    def generate_camera_poses(self, G):
        '''
        Calculate the camera poses from the video
        '''
        total_frames, new_camera_mtx, roi, cap = self.get_video_info_and_camera_matrix()
        camera_poses = []
        reference_board_id = None
        per_frame_board_points = {}
        
        for frame_idx in tqdm(range(total_frames), desc="Calculating Camera Poses"):
            ret, dist_frame = cap.read()
            if not ret:
                raise ValueError(f"Error reading frame {frame_idx} from video: {self.video}")
            
            frame = undistort_frame(dist_frame, self.camera_matrix, self.dist_coeffs, new_camera_mtx, roi)
            
            dot_images, poses, detected_boards, reprojection_error, board_markers_info = detect_pose(
                frame, new_camera_mtx, None, self.boards, self.dictionary, plot=True
            )
            if poses:
            
                valid_board_ids, rvecs, tvecs, reproj_er, board_points, board_point_pairs = get_valid_boards(
                    G, detected_boards, poses, reprojection_error, board_markers_info
                )
               
                if not valid_board_ids:
                    print(f"No valid board instance found in graph G at frame {frame_idx}.")
                    continue

                if reference_board_id is None:
                    reference_board_id = get_root_board_index(G)

                reproj_er = [ v if v is not None else float("inf") for v in reproj_er]
                
                # Use the multiboard PnP function
                world_T_camera = multiboard_pnp(
                    G, valid_board_ids, board_points, new_camera_mtx,  None, len(valid_board_ids), reference_board_id, self.snap
                )            
                
                if world_T_camera is None or len(valid_board_ids) < 3:
                    camera_poses, per_frame_board_points = self.add_empty_pose(camera_poses, frame_idx, per_frame_board_points)
                    continue
                
                # Camera Poses in TUM Format: timestamp x y z q_x q_y q_z q_w 
                camera_poses.append(format_pose(frame_idx, world_T_camera))
                if frame_idx not in per_frame_board_points:
                    per_frame_board_points[frame_idx] = []
                per_frame_board_points[frame_idx].append((len(valid_board_ids), board_point_pairs))
                valid_pose_found = True
                
            else:
                camera_poses, per_frame_board_points = self.add_empty_pose(camera_poses, frame_idx, per_frame_board_points)
        cap.release()

        camera_poses_path = get_result_save_path(self.video, result_type="camera_poses", extension=".txt",  output_path = self.output_path)
        
        traj = parse_trajectory(camera_poses)
        save_trajectory_evo(traj, camera_poses_path)
        plot_trajectory_evo(traj,camera_poses_path.replace(".txt", ".png"))
        
        # plot_trajectory_2D(camera_poses_path, camera_poses)
        
        per_frame_board_points_path = get_result_save_path(self.video, result_type="detected_points",filename_prefix="detected_points_", extension=".json",  output_path = self.output_path)
        save_board_points_to_json(per_frame_board_points, per_frame_board_points_path)


    def bundle_pnp_solver(self, G):
        # Input 1:  Serialize Camera Intrinsics
        new_camera_mtx= self.get_video_info_and_camera_matrix()[1]
        camera_matrix_str, dist_coeffs_str = serialize_intrinsics(new_camera_mtx, None)

        # Input 2: Camera Trajectory Path
        camera_trajectory = get_result_save_path(self.video, result_type="camera_poses", extension=".txt",  output_path = self.output_path)

        # Input 3: Graph poses
        board_data = extract_poses(G)
        board_data_str = ";".join(board_data)

        # Input 4: Pose Graph Camera Trajectory Path
        graph_camera_trajectory, replacement, filename = build_pose_graph_camera_trajectory_path(self.use_pose_graph, camera_trajectory)

        # Input 5: Detected Points
        detected_points_camera = get_result_save_path(self.video, result_type="detected_points", filename_prefix="detected_points_", extension=".json",  output_path = self.output_path)

        # Input 6: Detected points For Graph Close Camera Trajectory 
        detected_path_graph = os.path.dirname(detected_points_camera.replace("gt_view", "pose_graph_extra_frames"))
        detected_points_graph = os.path.join(detected_path_graph, f"detected_points_{filename}.json")
        detected_points_graph_segments = detected_points_graph.split(os.sep)
        idx = detected_points_graph_segments.index("pose_graph_extra_frames")
        detected_points_graph_segments[idx - 1] = replacement
        detected_points_graph = os.sep.join(detected_points_graph_segments)

        # Input 7: Camera Intrinsics for Close View
        segments = self.video.split(os.sep)
        bench_idx = segments.index("Benchmark")
        base_path = os.sep.join(segments[:bench_idx + 1])
        intrinsics_path = os.path.join(base_path, replacement, "pose_graph_extra_frames", "intrinsics")
        camera_matrix_close = np.load(os.path.join(intrinsics_path, "camera_mtx.npy"))
        dist_coeffs_close = np.load(os.path.join(intrinsics_path, "camera_dist.npy"))
        camera_matrix_close, _ = get_undistorted_camera_matrix(camera_matrix_close, dist_coeffs_close, 2160, 3840)
        camera_matrix_close_str, dist_coeffs_close_str = serialize_intrinsics(camera_matrix_close, None)

        cpp_program_path = "princeton365/optimization/bundle_pnp/build/board_experiment"
        command = [cpp_program_path, camera_matrix_str, dist_coeffs_str, camera_trajectory, board_data_str, graph_camera_trajectory, detected_points_camera, detected_points_graph, camera_matrix_close_str, dist_coeffs_close_str]
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

        camera_poses_path = get_result_save_path(self.video, result_type="camera_poses", extension=".txt",  output_path = self.output_path)
        traj = file_interface.read_tum_trajectory_file(camera_poses_path)
        plot_trajectory_evo(traj,camera_poses_path.replace(".txt", ".png"))
        

    def run(self):
        if self.use_pose_graph:
            G = load_graph(self.use_pose_graph)
        else:
            G = self.generate_pose_graph()
        
        G = convert_multidigraph_to_digraph(G)
        G = pose_graph_optimization(G)
    
        if self.snap:
            G = snap_coordinate_systems(G)
        
        if not self.skip_camera_poses:
            self.generate_camera_poses(G)

        if self.bundle_pnp:
            self.bundle_pnp_solver(G)


        

def main():
    parser = argparse.ArgumentParser(description="Generate and optionally save or view Charuco/Grid boards")
    
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to input video file")
    parser.add_argument("--intrinsics", type=str, default="intrinsics", help="Path to intrinsics folder") 
    parser.add_argument("--use_pose_graph", type=str, default = "", help="Path to pose graph pickle file")
    parser.add_argument("--snap", action="store_true", help="Snap the pose graph")
    parser.add_argument("--skip_camera_poses", action="store_true", help="Skip calculation of camera poses")
    parser.add_argument("--bundle_pnp", action="store_true", help="Bundle adjustment using PnP")
    parser.add_argument("--config", type=str, default = "princeton365/configs/board_configs.yaml", help="Path to YAML config file for board generation")
    parser.add_argument("--board_type", type=str, default="grid", help="Either 'charuco' or 'grid'")
    parser.add_argument("--output_path", type=str, default = None, help="Path to save the outputs")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")

    args = parser.parse_args()
    
    pipeline = Princeton365(args)
    pipeline.run()

if __name__ == "__main__":
    main()
