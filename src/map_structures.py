# vo_project/map_structures.py

import numpy as np
import open3d as o3d
from dataclasses import dataclass

@dataclass
class MapPoint:
    """Represents a 3D point in the map."""
    id: int
    position: np.ndarray  # 3x1 vector
    observations: list   # List of (keyframe_id, keypoint_index)
    color: np.ndarray      # 3x1 vector (normalized RGB)

@dataclass
class Keyframe:
    """Represents a keyframe with its pose, features, and observations."""
    id: int
    R: np.ndarray        # 3x3 rotation matrix (world to camera)
    t: np.ndarray        # 3x1 translation vector (camera position in world)
    keypoints: list
    descriptors: np.ndarray
    observations: list   # List of (map_point_id, keypoint_index)
    img: np.ndarray      # The RGB image for visualization

class Map:
    """
    Represents the full map, managing keyframes and 3D map points.
    
    Attributes:
        keyframes (dict): Maps keyframe IDs to Keyframe objects.
        map_points (dict): Maps map point IDs to MapPoint objects.
        next_keyframe_id (int): Counter for assigning unique keyframe IDs.
        next_map_point_id (int): Counter for assigning unique map point IDs.
    """
    def __init__(self):
        self.keyframes = {}
        self.map_points = {}
        self.next_keyframe_id = 0
        self.next_map_point_id = 0

    def add_keyframe(self, keyframe: Keyframe):
        """Adds a keyframe to the map."""
        if keyframe.id in self.keyframes:
            raise ValueError(f"Keyframe with ID {keyframe.id} already exists.")
        self.keyframes[keyframe.id] = keyframe
        self.next_keyframe_id += 1

    def add_map_point(self, map_point: MapPoint):
        """Adds a map point to the map."""
        if map_point.id in self.map_points:
            raise ValueError(f"MapPoint with ID {map_point.id} already exists.")
        self.map_points[map_point.id] = map_point
        self.next_map_point_id += 1
    
    def get_pcd(self):
        """Generates an Open3D point cloud from the map points."""
        positions = [p.position for p in self.map_points.values()]
        colors = [p.color for p in self.map_points.values()]

        if not positions:
            return o3d.geometry.PointCloud()
        
        # Convert the list of arrays into a single (N, 3) NumPy array.
        # CRITICAL FIX: Ensure the data type is float64 for Open3D.
        points_np = np.squeeze(np.array(positions)).astype(np.float64)
        colors_np = np.squeeze(np.array(colors)).astype(np.float64)

        # The np.squeeze operation can remove a dimension for a single point,
        # so we reshape it back to (1, 3) if that happens.
        if points_np.ndim == 1:
            points_np = points_np.reshape(1, -1)
            colors_np = colors_np.reshape(1, -1)
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        return pcd