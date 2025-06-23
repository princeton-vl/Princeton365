#!/usr/bin/env python3
"""
Convert OpenCV intrinsics + per-frame camera->world poses (in QW,QX,QY,QZ order)
to a COLMAP text model using the OPENCV camera model, with stride support.

Writes the files in the output directory:
  {output_dir}/{poses_filename}/our_gt_to_user/
  1) cameras.txt
  2) images.txt
  3) points3D.txt (empty)

Any frames not a multiple of STRIDE are skipped.
"""

import os
import sys
import numpy as np
import cv2
import argparse
from scipy.spatial.transform import Rotation as R

def remove_frame_from_images(images_txt_path, frame_id):
    """
    Given an images.txt file (COLMAP format) and a frame id,
    remove the two lines corresponding to that frame. The first line
    starts with the numeric id, and the following line is the (empty)
    POINTS2D line.
    """
    frame_id_str = str(frame_id)
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        # Check if the current line starts with the given id.
        tokens = lines[i].strip().split()
        if tokens and tokens[0] == frame_id_str:
            # Skip this line and the next one.
            i += 2
        else:
            new_lines.append(lines[i])
            i += 1

    with open(images_txt_path, "w") as f:
        f.writelines(new_lines)

def main():
    parser = argparse.ArgumentParser(description="Convert OpenCV intrinsics and per-frame camera->world poses to a COLMAP text model using the OPENCV camera model.")
    
    parser.add_argument("--trajectory", type=str, default=None, required=True, help="Path to the trajectory file.")
    parser.add_argument("--intrinsics_dir", type=str, required=True, help="Directory containing the camera intrinsics files (camera_mtx.npy and camera_dist.npy).")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Directory containing the COLMAP outputs for the specific trajectory.")
    parser.add_argument("--gt_rotation", type=str, default=None, required=False, help="Path to the ground truth rotation file (T_gt.npy). If provided, it will be used to rotate the trajectory.")
    parser.add_argument("--resolution", default=(3840,2160), help="Resolution of the video (default: 3840x2160).")
    parser.add_argument("--stride", type=int, default=10, help="Process only frames that are multiples of this stride (default: 10)")
    parser.add_argument("--output_dir", type=str, default="results/colmap", help="Directory to save the COLMAP output files (default: results/colmap).")
    args = parser.parse_args()

    poses_filename = base_name = os.path.splitext(os.path.basename(trajectory_path))[0]
    
    # Locate relative_gt_to_user.npy in the same directory as the found video file.
    relative_gt_path = os.path.join(os.path.dirname(args.intrinsics_dir), "relative_gt_to_user_optim.npy")
    if not os.path.isfile(relative_gt_path):
        sys.exit(f"Error: {relative_gt_path} not found in video directory")
        
    print("Relative GT file found at:", relative_gt_path)
    relative_pose = np.load(relative_gt_path)
    if args.gt_rotation is not None:
        T_gt = np.load(args.gt_rotation)
        relative_pose = np.linalg.inv(T_gt) @ relative_pose
        
    # Intrinsics files are in the provided intrinsics_dir
    camera_mtx_file = os.path.join(args.intrinsics_dir, "camera_mtx.npy")
    camera_dist_file = os.path.join(args.intrinsics_dir, "camera_dist.npy")
    if not os.path.isfile(camera_mtx_file) or not os.path.isfile(camera_dist_file):
        sys.exit("Error: Could not find camera intrinsics files in the provided intrinsics_dir.")
        
    # The output directory is:
    check = os.path.join(args.output_dir, poses_filename)
    output_dir = os.path.join(args.output_dir, poses_filename, "our_gt_to_user")
    os.makedirs(output_dir, exist_ok=True)
    stride = args.stride

    # Load intrinsics
    mtx = np.load(camera_mtx_file)  # shape (3, 3)
    dist = np.load(camera_dist_file)  # shape (1, N) or (N,)

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    # In case dist has more than 4 parameters, take only the first four.
    k1, k2, p1, p2 = dist.flatten()[:4]

    width = args.resolution[0]
    height = args.resolution[1]

    # 1) Write cameras.txt
    cameras_txt_path = os.path.join(output_dir, "cameras.txt")
    camera_id = 1
    with open(cameras_txt_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{camera_id} OPENCV {width} {height} "
                f"{fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")

    # 2) Write images.txt
    images_txt_path = os.path.join(output_dir, "images.txt")
    with open(images_txt_path, "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        image_id = 1
        with open(args.trajectory, "r") as pose_file:
            for line in pose_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                frame_idx = int(float(parts[0]))
                # The file has camera->world transforms: Tcw, Qcw
                tx = float(parts[1])
                ty = float(parts[2])
                tz = float(parts[3])
                qx_cw = float(parts[4])
                qy_cw = float(parts[5])
                qz_cw = float(parts[6])
                qw_cw = float(parts[7])

                # Skip frames not matching the stride
                if frame_idx % stride != 0:
                    continue

                if any(np.isnan(val) for val in [tx, ty, tz, qw_cw, qx_cw, qy_cw, qz_cw]):
                    print(f"NaN detected in frame {frame_idx}: {tx}, {ty}, {tz}, {qw_cw}, {qx_cw}, {qy_cw}, {qz_cw}")
                    images_txt_path =  f'{args.colmap_dir}/images.txt'
                    remove_frame_from_images(images_txt_path, image_id)
                    image_id += 1
                    continue
                
                # Invert the pose to world->camera for COLMAP.
                Tcw = np.eye(4)
                
                # 1) Get Rcw from the quaternion (camera->world)
                Rcw = R.from_quat([qx_cw, qy_cw, qz_cw, qw_cw]).as_matrix()
                Tcw[:3, :3] = Rcw
                Tcw[:3, 3] = [tx, ty, tz]

                new_Tcw = Tcw @ relative_pose
                Rcw = new_Tcw[:3, :3]
                Tcw = new_Tcw[:3, 3]

                # 2) Compute Rwc = Rcw^T
                Rwc = Rcw.T

                # 3) Camera center in world is (tx, ty, tz), so t_wc = -R_wc * T_cw
                twc = -Rwc @ Tcw

                # 4) Compute quaternion for Rwc: inverse of a unit quaternion (negate vector part)
                q_wc = R.from_matrix(Rwc).as_quat()
                qw_wc, qx_wc, qy_wc, qz_wc = q_wc[3], q_wc[0], q_wc[1], q_wc[2]
              
                # Write to images.txt in COLMAP format:
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                image_name = f"frame_{frame_idx:06d}.jpg"
                f_img.write(f"{image_id} {qw_wc} {qx_wc} {qy_wc} {qz_wc} "
                            f"{twc[0]} {twc[1]} {twc[2]} {camera_id} {image_name}\n")
                
                # Empty line for POINTS2D
                f_img.write("\n")
                image_id += 1


    # 3) Write an empty points3D.txt
    points3D_txt_path = os.path.join(output_dir, "points3D.txt")
    with open(points3D_txt_path, "w") as f_pts:
        f_pts.write("# 3D point list with one line of data per point:\n")
        f_pts.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f_pts.write("# Number of points: 0, mean track length: 0\n")

if __name__ == "__main__":
    main()
