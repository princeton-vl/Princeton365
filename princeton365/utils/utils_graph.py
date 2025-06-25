import os 
import cv2
import types
import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import networkx as nx


def update_edge_transformations(G):
    '''
    Update the edge transformations in the graph based on the node poses.
    '''
    for u, v in G.edges():
        pose_u = G.nodes[u]["pose"]
        pose_v = G.nodes[v]["pose"]

        T = np.linalg.inv(pose_u) @ pose_v
        distance = round(np.linalg.norm(T[:3, 3]), 3)

        G[u][v]["transformation"] = T
        G[u][v]["distance"] = distance

        if G.has_edge(v, u):
            G[v][u]["transformation"] = np.linalg.inv(T)
            G[v][u]["distance"] = distance

    return G

def snap_coordinate_systems(G, plane = 2):
    '''
    Align the local Z-axis of each node with that of the reference node.
    '''
    nodes = list(G.nodes())
    if not nodes:
        return G

    reference_node = nodes[0]
    reference_pose = G.nodes[reference_node]["pose"]

    for idx in range(1, len(nodes)):
        current_node = nodes[idx]
        current_pose = G.nodes[current_node]["pose"]

        R_ref = reference_pose[:3, :3]
        t_ref = reference_pose[:3, 3]
        R_curr = current_pose[:3, :3]
        t_curr = current_pose[:3, 3]

        # Align the Z axis of current_pose with the Z axis of reference_pose
        z_ref = R_ref[:, 2]
        z_curr = R_curr[:, 2]

        v = np.cross(z_curr, z_ref)
        s = np.linalg.norm(v)
        c = np.dot(z_curr, z_ref)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        if s != 0:
            R_align_z = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
        else:
            R_align_z = np.eye(3)

        R_new = R_align_z @ R_curr

        t_new = t_curr.copy()
        t_new[plane] = t_ref[plane]

        new_pose = np.eye(4)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = t_new
        G.nodes[current_node]["pose"] = new_pose

        reference_node = current_node
        reference_pose = new_pose

    G = update_edge_transformations(G)

    return G


def median_edge_selection(G):
    '''
    Selects the median distance for each edge in the graph G and return the ege closest to that median.
    '''
    processed_pairs = set()

    for u, v, data in G.edges(data=True):
        if (u, v) in processed_pairs:
            continue
        processed_pairs.add((u, v))
        
        # Collect all distances involving the nodes u and v
        distances = [d['distance'] for x, y, d in G.edges(data=True) if (x == u and y == v)]

        median = statistics.median(distances)
        print(f"Median distance for edge ({u}, {v}): {median}")
        specific_edges = [(x, y, d) for x, y, d in G.edges(data=True) if (x == u and y == v)]
        closest_edge = min(
            specific_edges, 
            key=lambda x: abs(x[2]['distance'] - median)
        )
        closest_distance = closest_edge[2]['distance']
        closest_transformation = closest_edge[2]['transformation']

        for x, y, d in G.edges(data=True):
            if (x == u and y == v):
                d['transformation'] = closest_transformation
                d['distance'] = closest_distance
    return G

def convert_multidigraph_to_digraph(G):
    G_new = nx.DiGraph()
    G_new.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if G_new.has_edge(u, v):
            continue
        else:
            G_new.add_edge(u, v, **data)
    return G_new


def load_graph(path):
    '''
    Loads a graph from a pickle file.
    '''
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded from {path}")
    return G

def save_graph(G, path):
    '''
    Saves a graph to a pickle file.
    '''
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")

def plot_graph(G, title=None, edge_labels=None, save_path=None):
    '''
    Plots the graph G
    '''
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw( 
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=10,
        font_weight="bold",
        edge_color="gray" if edge_labels else "black",  # Gray if drawing edges
    )
    plt.title(title)
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_root_board_index(G):
    """
    Returns the node with identity pose (used as reference).
    """
    for node, data in G.nodes(data=True):
        if np.array_equal(data.get("pose"), np.eye(4)):
            return node
    return None 


def extract_poses(G):
    '''
    Extracts the poses of the boards in the graph G.
    '''
    board_data = []

    for node in G.nodes:
        node_id_int = int(node.split("_")[0])

        pose_matrix = G.nodes[node]['pose']  
        translation = pose_matrix[:3, 3] 
        
        # Extract rotation (angle-axis representation)
        rotation_matrix = pose_matrix[:3, :3]
        angle_axis = R.from_matrix(rotation_matrix).as_rotvec()
        pose_str = ",".join(map(str, [*angle_axis, *translation]))
        board_data.append(f"{node_id_int}:{pose_str}")
        
    return board_data


def get_valid_boards(G, detected_boards, poses, reprojection_error, board_markers_info):
    '''
    Returns the valid boards, their rotation and translation vectors, reprojection errors,
    and the 3D-2D point pairs.
    '''
    valid_board_ids, rvecs, tvecs, reproj_errors = [], [], [], []
    board_points = {"3d": [], "2d": []}
    board_point_pairs = []
    
    for idx, board_index in enumerate(detected_boards):
        board_id = f"{board_index}_0"
        if board_id in G and idx < len(poses):
            valid_board_ids.append(board_id)
            rvec, tvec = poses[idx]
            rvecs.append(rvec)
            tvecs.append(tvec)
            reproj_errors.append(reprojection_error[idx])
            board_points["3d"].append(board_markers_info["3d"][idx])
            board_points["2d"].append(board_markers_info["2d"][idx])
            board_point_pairs.append([board_id, board_markers_info["3d"][idx], board_markers_info["2d"][idx]])
        else:
            print(f"Board {board_id} not found in graph.")
    
    return valid_board_ids, rvecs, tvecs, reproj_errors, board_points, board_point_pairs
