import g2o
import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd


class PoseGraphOptimization:
    '''
    Class to perform pose graph optimization using g2o.
    It initializes the optimizer, adds nodes and edges, and performs optimization.
    The class also provides a method to retrieve the optimized poses.
    '''
    def __init__(self):
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)
        self.node_id = "0_0"
        self.nodes = {}

    def add_node(self, pose):
        vertex = g2o.VertexSE3()
        vertex.set_id(int(self.node_id.split("_")[0]))
        vertex.set_estimate(pose)
        if self.node_id == "0_0":
            vertex.set_fixed(True)  # Fix the first node as the reference
        self.optimizer.add_vertex(vertex)
        self.nodes[self.node_id] = vertex
        self.node_id = f"{int(self.node_id.split('_')[0])+1}_0"
        return vertex

    def add_edge(
        self, vertex_from, vertex_to, relative_pose, information=np.identity(6)
    ):
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, vertex_from)
        edge.set_vertex(1, vertex_to)
        edge.set_measurement(relative_pose)
        edge.set_information(information)
        self.optimizer.add_edge(edge)

    def optimize(self, iterations=10):
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(iterations)

    def get_poses(self):
        poses = {}
        for i in self.nodes:
            logging.info(f"Estimating pose for node {i}")
            poses[i] = self.nodes[i].estimate()
        return poses


def relative_pose_to_g2o(matrix):

    t = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = g2o.Quaternion(rotation_matrix)
    return g2o.Isometry3d(quaternion, t)


def pose_graph_optimization(G):
    pgo = PoseGraphOptimization()
    nodes = {}
    nx_to_pgo_mapping = {}

    # Add nodes to the optimizer
    for idx, node in enumerate(G.nodes):
        pose = G.nodes[node]["pose"]
        g2o_pose = relative_pose_to_g2o(pose)
        pgo_node = pgo.add_node(g2o_pose)
        nodes[node] = pgo_node
        nx_to_pgo_mapping[node] = f"{idx}_0"

    for u, v, data in G.edges(data=True):
        relative_pose = data["transformation"]
        g2o_relative_pose = relative_pose_to_g2o(relative_pose)
        pgo.add_edge(nodes[u], nodes[v], g2o_relative_pose)

    # Optimize the pose graph
    pgo.optimize()

    # Get the optimized poses and update the graph
    optimized_poses = pgo.get_poses()
    for node in G.nodes:
        node_id = nx_to_pgo_mapping[node]
        optimized_pose = optimized_poses[node_id]
        G.nodes[node]["pose"] = optimized_pose.matrix()

    for u, v in G.edges():
        pose_u = optimized_poses[nx_to_pgo_mapping[u]]
        pose_v = optimized_poses[nx_to_pgo_mapping[v]]

        # Calculate the new relative transformation
        T_u = pose_u.matrix()
        T_v = pose_v.matrix()
        T_relative = np.linalg.inv(T_u) @ T_v

        # Calculate the distance
        distance = round(np.linalg.norm(T_relative[:3, 3]), 3)

        # Update the edges in both directions
        G.edges[u, v]["transformation"][:3, :3] = T_relative[:3, :3]
        G.edges[u, v]["distance"] = distance
        G.edges[v, u]["transformation"][:3, :3] = np.linalg.inv(T_relative)[:3, :3]
        G.edges[v, u]["distance"] = distance
    return G
