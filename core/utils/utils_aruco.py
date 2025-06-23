import os 
import cv2
import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx



def detect_pose(image, camera_matrix, dist_coeffs, boards, dictionary, plot=False):
    
    detected_boards = []
    undistorted_image = image
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 100
    params.cornerRefinementMinAccuracy = 0.001

    params.minMarkerPerimeterRate = 0.002  # Lower this for smaller markers

    poses = []
    reprojection_errors = []
    board_markers_info = {"3d": [], "2d": []}

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        undistorted_image, dictionary, parameters=params
    )

    if marker_ids is not None and len(marker_ids) > 0:
        for i, board in enumerate(boards):
            detector.refineDetectedMarkers(
                image, board, marker_corners, marker_ids, rejected, camera_matrix, dist_coeffs
            )
            objPoints = []
            imgPoints = []
            if isinstance(board, cv2.aruco.CharucoBoard):
                current_ids = [mid for mid in marker_ids if mid[0] in board.getIds()]
                current_corners = [
                    marker_corners[i]
                    for i, mid in enumerate(marker_ids)
                    if mid[0] in board.getIds()
                ]
                current_ids = np.array(current_ids)
                current_corners = np.array(current_corners)
                if len(current_ids) > 0:
                    charuco_retval, charuco_corners, charuco_ids = (
                        cv2.aruco.interpolateCornersCharuco(
                            current_corners, current_ids, undistorted_image, board
                        )
                    )

                    if charuco_retval:
                        objPoints, imgPoints = matchImagePoints(
                            board, charuco_corners, charuco_ids
                        )
            elif isinstance(board, cv2.aruco.GridBoard):
                filtered_marker_corners = []
                filtered_marker_ids = []
                board_ids = board.getIds().flatten()
                board_obj_points = board.getObjPoints()
                for j, marker_id in enumerate(marker_ids):
                    if marker_id[0] in board_ids:
                        filtered_marker_corners.append(marker_corners[j])
                        filtered_marker_ids.append(marker_id)
                        marker_corners_current = marker_corners[j]
                        marker_index = np.where(board_ids == marker_id[0])[0][0]
                        marker_obj_points = board_obj_points[marker_index]
                        objPoints.extend(marker_obj_points)
                        imgPoints.extend(marker_corners_current.reshape(-1, 2))
             
            if ((len(objPoints) >= 4 and isinstance(board, cv2.aruco.CharucoBoard)) or (len(objPoints) > 25 and isinstance(board, cv2.aruco.GridBoard))) and len(imgPoints) == len(objPoints):
                objPoints = np.array(objPoints, dtype=np.float32).reshape(-1, 3)
                unique_x = np.unique(objPoints[:, 0])
                unique_y = np.unique(objPoints[:, 1])
                if len(unique_x) == 1 or len(unique_y) == 1:
                    retval = False
                else:
                    try:
                        if isinstance(board, cv2.aruco.GridBoard):
                            objPoints = np.array(objPoints, dtype=np.float32)  # 3D points for the object
                            imgPoints = np.array(imgPoints, dtype=np.float32)  # 2D points for the imag
                            if len(imgPoints.shape) == 2:  # If imgPoints is (N, 2)
                                imgPoints = imgPoints[:, np.newaxis, :]
                            homography, mask = cv2.findHomography(objPoints[:, :2], imgPoints, cv2.RANSAC, ransacReprojThreshold=19.0)
                        else: 
                            homography, mask = cv2.findHomography(objPoints[:, :2], imgPoints, cv2.RANSAC, ransacReprojThreshold=3.0)
                        retval, rotations, translations, normals = cv2.decomposeHomographyMat(
                            homography, camera_matrix
                        )
                        rvec, tvec = cv2.Rodrigues(rotations[0])[0], translations[0]
                        
                        inliers = mask.ravel().astype(bool)  # Convert mask to a boolean array
                        imgPoints = imgPoints[inliers]
                        objPoints = objPoints[inliers]
                        retval, rvec, tvec = cv2.solvePnP(
                            objPoints,
                            imgPoints,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_IPPE

                        )
    
                    except cv2.error as e:
                        retval = False
                        print("Error in solvePnP:", e)
            else:
                retval = False
            if retval:
                projected, _ = cv2.projectPoints(
                    objPoints, rvec, tvec, camera_matrix, dist_coeffs
                )  # Jacobian shape is 2N x (10+Num of distortion coefficients)
                
                initial_error = np.linalg.norm(projected - imgPoints)

                if ((len(objPoints) >= 4 and isinstance(board, cv2.aruco.CharucoBoard)) or (len(objPoints) > 25 and isinstance(board, cv2.aruco.GridBoard))): 
                    optimized_rvec, optimized_tvec = cv2.solvePnPRefineLM(
                        objPoints, imgPoints, camera_matrix, dist_coeffs, rvec, tvec
                    )

                    # Recalculate reprojection error after optimization
                    optimized_projected, _ = cv2.projectPoints(
                        objPoints,
                        optimized_rvec,
                        optimized_tvec,
                        camera_matrix,
                        dist_coeffs,
                    )
  
                    errors_per_point = np.linalg.norm(optimized_projected.squeeze() - imgPoints.squeeze(), axis=1)
                    threshold = 10.0 if isinstance(board, cv2.aruco.GridBoard) else 2.0
                    valid_indices = np.where(errors_per_point <= threshold)[0] 
        
                    optimized_projected = optimized_projected[valid_indices]
                    optimized_error = np.linalg.norm(
                        optimized_projected - imgPoints[valid_indices]
                    )
                    
                    plot_flag = False
                    is_optimized =False

                    if optimized_error < initial_error:
                        if optimized_error < 15.0 and isinstance(board, cv2.aruco.CharucoBoard):
                            objPoints = objPoints[valid_indices]
                            imgPoints = imgPoints[valid_indices]
                            is_optimized = True
                            plot_flag = True
                            detected_boards.append(i)
                            rvec, tvec = optimized_rvec, optimized_tvec
                            poses.append((optimized_rvec, optimized_tvec))
                            reprojection_errors.append(optimized_error)
                            board_markers_info["3d"].append(objPoints)
                            board_markers_info["2d"].append(imgPoints)

                        else: 
                            objPoints = objPoints[valid_indices]
                            imgPoints = imgPoints[valid_indices]
                            is_optimized = True
                            plot_flag = True
                            detected_boards.append(i)
                            rvec, tvec = optimized_rvec, optimized_tvec
                            poses.append((optimized_rvec, optimized_tvec))
                            reprojection_errors.append(optimized_error)
                            board_markers_info["3d"].append(objPoints)
                            board_markers_info["2d"].append(imgPoints)

                    else:
                        if initial_error < 15.0 and isinstance(board, cv2.aruco.CharucoBoard):
                            errors_per_point_init = np.linalg.norm(projected.squeeze() - imgPoints.squeeze(), axis=1)
                            threshold = 2.0
                            valid_indices_init = np.where(errors_per_point_init <= threshold)[0] 
                            objPoints = objPoints[valid_indices_init]
                            imgPoints = imgPoints[valid_indices_init]
                            projected = projected[valid_indices_init]
                            initial_error = np.linalg.norm(projected - imgPoints)
                        
                            plot_flag = True
                            detected_boards.append(i)
                            poses.append((rvec, tvec))
                            reprojection_errors.append(initial_error)
                            board_markers_info["3d"].append(objPoints)
                            board_markers_info["2d"].append(imgPoints)

                        else: 
                            errors_per_point_init = np.linalg.norm(projected.squeeze() - imgPoints.squeeze(), axis=1)
                            threshold = 2.0
                            valid_indices_init = np.where(errors_per_point_init <= threshold)[0] 
                            objPoints = objPoints[valid_indices_init]
                            imgPoints = imgPoints[valid_indices_init]
                            projected = projected[valid_indices_init]
                            initial_error = np.linalg.norm(projected - imgPoints)
                        
                            plot_flag = True
                            detected_boards.append(i)
                            poses.append((rvec, tvec))
                            reprojection_errors.append(initial_error)
                            board_markers_info["3d"].append(objPoints)
                            board_markers_info["2d"].append(imgPoints)
    
                    if plot and plot_flag:
                        length_of_axis = 0.1
                        cv2.drawFrameAxes(
                            undistorted_image,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            length_of_axis,
                        )
                        
                        if is_optimized:
                            valid = valid_indices
                        else:
                            valid = valid_indices_init
                        
                        if isinstance(board, cv2.aruco.CharucoBoard):
                            flat_corners = [corner[0].tolist() for corner in charuco_corners]
                        else:
                            marker_corners_list = [corner[0] for corner in filtered_marker_corners]
                            flat_corners = [list(corner) for marker in marker_corners_list for corner in marker]
                        for i in valid:
                            corner_tuple = (int(flat_corners[i][0]), int(flat_corners[i][1]))
                            cv2.circle(
                                undistorted_image,
                                corner_tuple,
                                radius=1,
                                color=(0, 0, 255),
                                thickness=6,
                            )
                        all_indices = np.arange(len(flat_corners))  # Create an array of all indices
                        mask_bool = mask.ravel().astype(bool)
                        filtered_indices = all_indices[mask_bool]  # Map mask to original indices
                        final_valid_indices = filtered_indices[valid]
                        remaining_indices = np.setdiff1d(all_indices, final_valid_indices)
                        for i in remaining_indices:
                            corner_tuple = (int(flat_corners[i][0]), int(flat_corners[i][1]))
                            cv2.circle(
                                undistorted_image,
                                corner_tuple,
                                radius=1,
                                color=(0, 255, 0),  # Green
                                thickness=6)

    return undistorted_image, poses, detected_boards, reprojection_errors, board_markers_info


def estimate_relative_pose(image, camera_matrix, dist_coeffs, boards, dictionary, pair_detected=False):
    """
    Estimate the relative pose between all the boards in the image.
    """
    _, poses, detected_boards, _, _ = detect_pose(
        image, camera_matrix, dist_coeffs, boards, dictionary
    )

    if len(poses) < 2 or len(detected_boards) != len(poses):
        if len(detected_boards) == 1 and len(detected_boards) == len(poses):
            return (
                [],
                pair_detected,
                detected_boards[0],
            )  # Not enough data to estimate any relative poses

    board_pairs = list(itertools.combinations(range(len(detected_boards)), 2))

    results = []
    for i, j in board_pairs:
        pair_detected = True
        board_pair = [detected_boards[i], detected_boards[j]]
        pose_pair = [poses[i], poses[j]]

        # Sort poses and detected_boards so that lower index comes first
        if board_pair[0] > board_pair[1]:
            pose_pair[0], pose_pair[1] = pose_pair[1], pose_pair[0]
            board_pair.sort()

        pose1, pose2 = pose_pair[0], pose_pair[1]

        R1, _ = cv2.Rodrigues(pose1[0])
        R2, _ = cv2.Rodrigues(pose2[0])

        T1 = np.hstack((R1, pose1[1]))
        T1 = np.vstack((T1, [0, 0, 0, 1]))

        T2 = np.hstack((R2, pose2[1]))
        T2 = np.vstack((T2, [0, 0, 0, 1]))

        T_relative = np.linalg.inv(T1) @ T2
        results.append((board_pair, T_relative))

    return results, pair_detected, None


def multiboard_pnp( G, valid_board_id, board_points, camera_matrix, dist_coeffs, count_valid, reference_board_id, snap):
    """
    Transform the local coordinate system of the board to the common coordinate system.
    """

    objpoints = []
    imgpoints = []

    first_board_index = valid_board_id[0]
    objpoints.append(board_points["3d"][0])
    imgpoints.append(board_points["2d"][0])
    
    for i in range(1, count_valid):
        board_idx = valid_board_id[i]
        shortest_path = nx.shortest_path(
            G, source=first_board_index, target=board_idx, weight="distance"
        )

        T_local_to_board = np.eye(4)

        for j in range(len(shortest_path) - 1):
            # edge_data = next(iter(G[shortest_path[j]][shortest_path[j + 1]].values()))
            edge_data = G[shortest_path[j]][shortest_path[j + 1]]
            T_local_to_board = np.dot(T_local_to_board, edge_data["transformation"])

        board_3d_points = board_points["3d"][i]

        # Transform the 3D points to the coordinate system of the first board
        ones = np.ones(
            (board_3d_points.shape[0], 1)
        )  # Create a column of ones for homogeneous coordinates
        board_3d_points_hom = np.hstack(
            [board_3d_points, ones]
        )  # Convert to homogeneous coordinates (Nx4)
        transformed_points_hom = (
            T_local_to_board @ board_3d_points_hom.T
        ).T  # Apply transformation
        transformed_points = transformed_points_hom[:, :3]  # Convert back to 3D (Nx3)


        objpoints.append(transformed_points)
        imgpoints.append(board_points["2d"][i])


    objpoints = np.vstack(objpoints).reshape(-1, 3)
    imgpoints = np.vstack(imgpoints).reshape(-1, 2)

    try:
        if snap:
            success, rvec, tvec = cv2.solvePnP(
                objpoints,
                imgpoints,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objpoints,
                imgpoints,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            
            print("PnP solution failed")
            return None
    except cv2.error as e:
        print(f"Error in solvePnP: {e}")
        return None

    projected, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)

    initial_error = np.linalg.norm(projected.reshape(-1, 2) - imgpoints)

    optimized_rvec, optimized_tvec = cv2.solvePnPRefineLM(
        objpoints, imgpoints, camera_matrix, dist_coeffs, rvec, tvec
    )

    # Recalculate reprojection error after optimization
    optimized_projected, _ = cv2.projectPoints(
        objpoints, optimized_rvec, optimized_tvec, camera_matrix, dist_coeffs
    )
    optimized_error = np.linalg.norm(optimized_projected.reshape(-1, 2) - imgpoints)

    if optimized_error < initial_error:
        rvec, tvec = optimized_rvec, optimized_tvec

    T_board_to_camera = np.linalg.inv(get_transformation_matrix(rvec, tvec))

    shortest_path = nx.shortest_path(
        G, source=reference_board_id, target=first_board_index, weight="distance"
    )
    T_world_to_board = np.eye(4)
    for j in range(len(shortest_path) - 1):
        edge_data = G[shortest_path[j]][shortest_path[j + 1]]
        T_world_to_board = np.dot(T_world_to_board, edge_data["transformation"])

    T_world_to_camera = np.dot(T_world_to_board, T_board_to_camera)
    return T_world_to_camera

def matchImagePoints(charuco_board, charuco_corners, charuco_ids):
    '''
    Match the image points to the object points based on the gridboard.
    '''
    objPoints = []
    imgPoints = []
    for i in range(0, len(charuco_ids)):
        index = charuco_ids[i]
        objPoints.append(charuco_board.getChessboardCorners()[index])
        objPoints[-1][0][1] = (
            charuco_board.getRightBottomCorner()[1] - objPoints[-1][0][1]
        )
        imgPoints.append(charuco_corners[i])
    return np.array(objPoints), np.array(imgPoints)

def get_transformation_matrix(rvec, tvec):
    '''
    Get the transformation matrix from rotation and translation vectors.
    '''
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def undistort_frame(frame, camera_matrix, dist_coeffs, new_camera_mtx, roi):
    '''
    Undistort the frame using the camera matrix and distortion coefficients.
    '''
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]
