import torch
import h5py
import argparse
import mplcursors
import os
import json
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from scipy import integrate
from evo.core import geometry
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from evo.core.trajectory import PoseTrajectory3D
from princeton365.utils import utils_depth as dd
from princeton365.utils.utils_trajectory import plot_trajectory_evo, align_rotations, align_positions, diagnose_trajectories, parse_trajectory, pad_trajectory_with_nans, read_trajectory_file
from princeton365.utils.utils_io import plot_flow_histogram, rotation_distance, estimate_similarity_transformation

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def compute_induced_optical_flow(traj_est_aligned, traj_gt_user, depth_path, camera_mtx, n_pixels=1000, resolution=(3840, 2160)):
    """
    Compute the average magnitude of induced optical flow due to trajectory misalignment
    using numerical integration over image pixels and depth distribution.
    
    Optimized to reuse memory and precompute PDF for speed, while keeping original structure.
    """
    depth_subdir = os.path.join(depth_path, 'depth')
    if not os.path.isdir(depth_subdir):
        raise ValueError(f"The specified depth path does not contain a 'depth' subdirectory: {depth_subdir}")
    
    depth_files = sorted([os.path.join(depth_subdir, f) for f in os.listdir(depth_subdir) if f.endswith('.h5')])
    if not depth_files:
        raise ValueError(f"No H5 depth files found in {depth_subdir}")
    
    depth_file = depth_files[0]
    
    print("Sampling depths from depth maps for distribution fitting...")
    depth_data = dd.read_zed_depth(depth_file)
    n_samples = min(100000, depth_data.size)
    
    results = dd.analyze_depth_distribution(depth_data, n_samples=n_samples)
    best_model = results['best_model']
    best_type = results['best_type']
    print(f"Best model for: {best_type} with {results['opt_n_gaussian'] if best_type == 'Gaussian' else results['opt_n_gamma']} components")
    
    # Determine integration bounds
    if best_type == "Gaussian":
        if hasattr(best_model, 'distributions'):
            min_depth = float('inf')
            max_depth = 0
            for dist in best_model.distributions:
                mean = dist.means[0].item()
                std = np.sqrt(dist.covs[0].item())
                min_depth = min(min_depth, max(0.1, mean - 4 * std))
                max_depth = max(max_depth, mean + 4 * std)
        else:
            mean = best_model.means[0].item()
            std = np.sqrt(best_model.covs[0].item())
            min_depth = max(0.1, mean - 4 * std)
            max_depth = mean + 4 * std
    else:  # Gamma
        if hasattr(best_model, 'distributions'):
            min_depth = float('inf')
            max_depth = 0
            for dist in best_model.distributions:
                shape = dist.shapes[0].item()
                rate = dist.rates[0].item()
                min_depth = min(min_depth, max(0.1, shape / rate / 20))
                max_depth = max(max_depth, shape / rate * 4)
        else:
            shape = best_model.shapes[0].item()
            rate = best_model.rates[0].item()
            min_depth = max(0.1, shape / rate / 20)
            max_depth = shape / rate * 4

    grid_size = int(np.sqrt(n_pixels))
    u_step = resolution[1] // grid_size
    v_step = resolution[0] // grid_size
    u_coords = np.arange(0, resolution[1], u_step)[:grid_size]
    v_coords = np.arange(0, resolution[0], v_step)[:grid_size]
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    u = u_grid.flatten()
    v = v_grid.flatten()

    fx = camera_mtx[0, 0]
    fy = camera_mtx[1, 1]
    cx = camera_mtx[0, 2]
    cy = camera_mtx[1, 2]
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy

    n_pixels_grid = len(u)
    n_frames = min(len(traj_est_aligned.positions_xyz), len(traj_gt_user.positions_xyz))

    # Precompute PDF interpolation
    lookup_d = np.linspace(min_depth, max_depth, 20480)
    log_pdf = best_model.log_probability(torch.tensor(lookup_d[:, None], dtype=torch.float32)).numpy().flatten()
    pdf_interp = lambda d: np.exp(np.interp(d, lookup_d, log_pdf))

    # Precompute unprojection directions
    points_2d = np.stack([x_normalized, y_normalized, np.ones_like(x_normalized)], axis=1)
    points_2d_h = np.hstack((points_2d, np.ones((n_pixels_grid, 1))))

    # Preallocated buffers
    u_est = np.empty(n_pixels_grid)
    v_est = np.empty(n_pixels_grid)
    flow_u = np.empty(n_pixels_grid)
    flow_v = np.empty(n_pixels_grid)
    flow_magnitude = np.empty(n_pixels_grid)
    zeros_buffer = np.zeros(n_pixels_grid)

    flows_per_frame = []
    total_flow = 0.0
    count = 0

    print("Computing induced optical flow via numerical integration...")
    for idx in tqdm(range(n_frames)):
        # Rigid transformations
        quat_gt = np.roll(traj_gt_user.orientations_quat_wxyz[idx], -1)
        quat_est = np.roll(traj_est_aligned.orientations_quat_wxyz[idx], -1)
        R_gt = R.from_quat(quat_gt).as_matrix()
        R_est = R.from_quat(quat_est).as_matrix()
        T_gt = np.eye(4); T_gt[:3, :3] = R_gt; T_gt[:3, 3] = traj_gt_user.positions_xyz[idx]
        T_est = np.eye(4); T_est[:3, :3] = R_est; T_est[:3, 3] = traj_est_aligned.positions_xyz[idx]
        T_rel = T_est @ np.linalg.inv(T_gt)

        def flow_integrand(depth):
            if depth <= 0:
                return zeros_buffer

            points_gt = points_2d * depth
            points_gt_h = np.hstack((points_gt, np.ones((n_pixels_grid, 1))))
            points_est_h = (T_rel @ points_gt_h.T).T
            points_est = points_est_h[:, :3] / points_est_h[:, 3:4]
            z = points_est[:, 2]
            valid = z > 1e-6

            u_est[:] = np.nan
            v_est[:] = np.nan
            u_est[valid] = (points_est[valid, 0] * fx) / z[valid] + cx
            v_est[valid] = (points_est[valid, 1] * fy) / z[valid] + cy

            flow_u[valid] = u_est[valid] - u[valid]
            flow_v[valid] = v_est[valid] - v[valid]
            flow_magnitude[:] = np.sqrt(flow_u**2 + flow_v**2)
            flow_magnitude[~valid] = 0.0

            return flow_magnitude * pdf_interp(depth)

        flow_magnitudes_per_pixel, _ = integrate.quad_vec(flow_integrand, min_depth, max_depth, epsabs=1e-3, epsrel=1e-3)

        valid_flow = ~np.isnan(flow_magnitudes_per_pixel) & ~np.isinf(flow_magnitudes_per_pixel) & (flow_magnitudes_per_pixel > 0)
        flows_per_frame.append(flow_magnitudes_per_pixel[valid_flow])
        total_flow += np.sum(flow_magnitudes_per_pixel[valid_flow])
        count += np.sum(valid_flow)

    avg_flow_magnitude = total_flow / count if count > 0 else 0.0
    print(f"Average Induced Optical Flow Magnitude: {avg_flow_magnitude:.4f} pixels")

    return avg_flow_magnitude, flows_per_frame


def calculate_flow_auc(flows_per_frame, resolution, output_path):
    """
    Calculate and plot the Area Under the Curve (AUC) for optical flow error based on the percentage of inliers.
    
    Args:
        flows_per_frame: List of arrays containing flow magnitudes per frame
        resolution: Tuple (width, height) of the camera resolution
        output_path: Path to save the AUC plot
        
    Returns:
        float: The normalized AUC value (between 0 and 1)
    """

    if not flows_per_frame or all(len(f) == 0 for f in flows_per_frame):
        print("No valid flow data to compute AUC. Skipping.")
        return 0.0 
    # Concatenate flow magnitudes from all frames
    flow_magnitudes = np.concatenate(flows_per_frame)
    
    # Calculate max threshold (τ_max) = sqrt(width^2 + height^2)
    tau_max = 100 #np.sqrt(resolution[0]**2 + resolution[1]**2)
    
    thresholds = np.arange(0, tau_max, 0.01)
    inlier_fractions = []
    for tau in thresholds:
        P = np.mean(flow_magnitudes < tau)
        inlier_fractions.append(P)
    inlier_fractions = np.array(inlier_fractions)
    
    auc = np.trapezoid(inlier_fractions, thresholds)
    normalized_auc = auc / tau_max
    
    
    max_plot_points = 1000
    if len(thresholds) > max_plot_points:
        idxs = np.linspace(0, len(thresholds) - 1, max_plot_points).astype(int)
        thresholds_plot = thresholds[idxs]
        inliers_plot = inlier_fractions[idxs]
    else:
        thresholds_plot = thresholds
        inliers_plot = inlier_fractions

    plt.figure(figsize=(10, 6), dpi = 100)
    plt.plot(thresholds_plot, inliers_plot, 'b-', linewidth=2)
    plt.xlabel('Threshold (pixels)')
    plt.ylabel('Fraction of Inliers')
    plt.title(f'Optical Flow Error: Inlier Fraction vs. Threshold')
    plt.grid(alpha=0.3)
    
    plt.gca().text(
        0.6, 0.2,
        f'AUC: {auc:.2f}\nNormalized AUC: {normalized_auc:.4f}',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes 
    )
    # plt.subplots_adjust(bottom=0.15, top=0.9)
    plt.savefig(output_path)
    print(f"Flow AUC curve saved to {output_path}")
    print(f"Normalized AUC: {normalized_auc:.4f} (Raw AUC: {auc:.2f}, Max possible AUC: {tau_max:.2f})")
    
    return auc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path', type=str, required=True, help='The file path for the gt TXT trajectory (60 fps).')
    parser.add_argument('--est_path', type=str, required=True, help='The file path for the estimated TXT trajectory (60 fps).')
    parser.add_argument('--user_to_defaultgt', type=str, required=True, help='The file path for the relative transformation from the ground truth to the user view.')
    parser.add_argument('--gt_to_defaultgt', type=str, help='The file path for the relative transformation from the ground truth to the default ground truth view.')
    parser.add_argument('--intrinsics_user', type=str, required=True, help='The file path for the camera intrinsics of the user view.')
    parser.add_argument('--depth', type=str, required=True, help='The directory path for the depth maps of the user view.')
    parser.add_argument('--n_pixels', type=int, default=1000, help='Number of pixels to sample per frame for optical flow computation.')
    parser.add_argument('--resolution', type=tuple, default=(3840, 2160), help='The resolution of the camera.')
    parser.add_argument('--output_folder', type=str, default='output', help='The folder to save the output files.')
    parser.add_argument('--align_evo', action='store_true', help='Whether to align the trajectories with evo function.')
    parser.add_argument('--exp_id', type=str, default ="",help='Name of exp_id')
    args = parser.parse_args()
    args.resolution = np.array(args.resolution)
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    
    output_img = f'{output_folder}/{args.exp_id}_trajectory.png'
    output_metrics_path = f'{output_folder}/{args.exp_id}_output_metrics_simple.json'  # To save the average flow
    output_dif_rot = f'{output_folder}/{args.exp_id}_dif_rotation.png'
    output_flow_hist = f'{output_folder}/{args.exp_id}_flow_histogram.png'  # New output for flow histogram
    output_flow_auc = f'{output_folder}/{args.exp_id}_flow_auc.png'  # New output for flow AUC curve

    camera_mtx = np.load(os.path.join(args.intrinsics_user, 'camera_mtx.npy'))
    dist_coeffs = np.load(os.path.join(args.intrinsics_user, 'camera_dist.npy'))
    
    T_user_to_defaultgt = np.load(args.user_to_defaultgt)
    if args.gt_to_defaultgt is not None:
        T_gt_to_defaultgt = np.load(args.gt_to_defaultgt)
    else:
        T_gt_to_defaultgt = np.eye(4)

    T_user_to_gt = np.linalg.inv(T_gt_to_defaultgt) @ T_user_to_defaultgt

    
    gt_timestamps, gt_positions_xyz, gt_orientations_quat_wxyz = read_trajectory_file(args.gt_path,  tum_format=True)
    est_timestamps, est_positions_xyz, est_orientations_wxyz = read_trajectory_file(args.est_path, tum_format=True)

    if len(gt_timestamps) != len(est_timestamps):
        print("The number of timestamps in the ground truth and estimated trajectories do not match.")
        print("Padding the estimated trajectory with NaN values for missing timestamps...")
        est_timestamps, est_positions_xyz, est_orientations_wxyz = pad_trajectory_with_nans(
            est_timestamps, est_positions_xyz, est_orientations_wxyz, gt_timestamps
        )   
    
    traj_gt_to_world = PoseTrajectory3D(positions_xyz=gt_positions_xyz, orientations_quat_wxyz=gt_orientations_quat_wxyz, timestamps=gt_timestamps)
    traj_est = PoseTrajectory3D(positions_xyz=est_positions_xyz, orientations_quat_wxyz=est_orientations_wxyz, timestamps=est_timestamps)
    
    valid_indices, total_valid_gt, valid_est_given_gt = diagnose_trajectories(traj_est, traj_gt_to_world)
    
    if np.sum(valid_indices) > 0:
        print("\nAttempting alignment with only valid points...")

        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est.positions_xyz[valid_indices],
            orientations_quat_wxyz=traj_est.orientations_quat_wxyz[valid_indices],
            timestamps=traj_est.timestamps[valid_indices]
        )
        traj_gt_to_world = PoseTrajectory3D(
            positions_xyz=traj_gt_to_world.positions_xyz[valid_indices],
            orientations_quat_wxyz=traj_gt_to_world.orientations_quat_wxyz[valid_indices],
            timestamps=traj_gt_to_world.timestamps[valid_indices]
        )

        traj_gt_to_world.transform(T_user_to_gt, right_mul = True)
        traj_user_to_world = traj_gt_to_world
        
        if args.align_evo:
            print("\nAligning trajectories with evo functions...")
            traj_est.align(traj_user_to_world, correct_scale=True)
            traj_est_aligned = traj_est
        else:
            print("\nAligning trajectories with our method...")
            traj_est_aligned, transformation= align_positions(traj_est, traj_user_to_world)
            traj_est_aligned, _ = align_rotations(traj_est_aligned, traj_user_to_world)

    else:
        raise ValueError("No valid corresponding points found for alignment. Please check the trajectories.")

    output_img_gt = f'{output_folder}/{args.exp_id}_gt_trajectory.png'
    output_img_est = f'{output_folder}/{args.exp_id}_est_trajectory.png'

    print(f"Saving the ground truth and estimated trajectories to {output_img_gt} and {output_img_est} respectively.")
    
    plot_trajectory_evo(traj_user_to_world, output_img_gt)
    plot_trajectory_evo(traj_est, output_img_est, title="Estimated Trajectory")

    print("\nRunning trajectory diagnostics...")


    ATEs = np.linalg.norm(traj_est_aligned.positions_xyz - traj_user_to_world.positions_xyz, axis=1)
    
    # Calculate the mean and standard deviation of the AP errors
    ate = np.nanmean(ATEs)
    std_ate = np.nanstd(ATEs)    
    plot_trajectory_evo(traj_user_to_world, output_img, title="Comparison", traj_est = traj_est_aligned, ape = ate, std = std_ate)

    # check rotation difference
    print("\nCalculating rotation differences...")
    rotation_differences = []
    
    for est_quat, gt_quat in zip(traj_est_aligned.orientations_quat_wxyz, traj_user_to_world.orientations_quat_wxyz):
        # Convert quaternions from WXYZ to XYZW format
        est_quat_xyzw = np.roll(est_quat, -1)
        gt_quat_xyzw = np.roll(gt_quat, -1)
        
        # Convert to rotation matrices
        R_est = R.from_quat(est_quat_xyzw).as_matrix()
        R_gt = R.from_quat(gt_quat_xyzw).as_matrix()
        
        # Calculate rotation difference in radians
        diff = rotation_distance(R_est, R_gt)
        rotation_differences.append(np.degrees(diff))  # Convert to degrees
    
    rotation_differences = np.array(rotation_differences)
    mean_rotation_error = np.mean(rotation_differences)
    std_rotation_error = np.std(rotation_differences)
    
    print(f"Mean rotation error: {mean_rotation_error:.2f}° (std: {std_rotation_error:.2f}°)")


    # Compute Induced Optical Flow
    avg_flow_magnitude, flows_per_frame = compute_induced_optical_flow(
        traj_est_aligned, 
        traj_user_to_world, 
        args.depth, 
        camera_mtx, 
        n_pixels=args.n_pixels, 
        resolution = args.resolution
    )
    
    # Plot flow histogram
    plot_flow_histogram(flows_per_frame, output_flow_hist)
    
    # Calculate and plot flow AUC
    flow_auc = calculate_flow_auc(flows_per_frame, args.resolution, output_flow_auc)
    
    coverage = valid_est_given_gt / total_valid_gt if total_valid_gt > 0 else 0.0
    coverage = coverage * 100  # Convert to percentage
    if coverage > 0 and flow_auc > 0:
        composite = 2 * (flow_auc * coverage) / (flow_auc + coverage)
    else:
        composite = 0.0

    output_metrics = {
        "ate": float(ate),
        "std_ate": float(std_ate),
        "mean_rotation_error": float(mean_rotation_error),
        "std_rotation_error": float(std_rotation_error),
        "avg_flow_magnitude": float(avg_flow_magnitude),
        "flow_auc": float(flow_auc),
        "total_valid_gt_frames": int(total_valid_gt),
        "valid_est_frames_given_valid_gt": int(valid_est_given_gt),
        "coverage": float(coverage),
        "composite": float(composite)
    }

    with open(output_metrics_path, 'w') as f:
        json.dump(output_metrics, f)

if __name__ == '__main__':
    main()

