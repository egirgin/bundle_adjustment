# vo_project/visualization.py

import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from map_structures import Map

def plot_and_save_sparsity(sparsity_matrix, fixed_kf_id, last_kf_id, output_dir):
    """Saves a plot of the Jacobian's sparsity structure."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.spy(sparsity_matrix, markersize=1)
    plt.title(f"Jacobian Sparsity (LBA for KF {fixed_kf_id} to {last_kf_id})")
    plt.xlabel("Parameters (Camera Poses + 3D Points)")
    plt.ylabel("Residuals (Observations)")
    plot_filename = os.path.join(output_dir, f"sparsity_{fixed_kf_id}.png")
    plt.savefig(plot_filename)
    plt.close()

def plot_and_save_trajectory_2d(gmap: Map, output_dir: str):
    """Generates a top-down (X-Z) plot of the camera's trajectory."""
    os.makedirs(output_dir, exist_ok=True)
    
    sorted_kf_ids = sorted(gmap.keyframes.keys())
    if len(sorted_kf_ids) < 2:
        return

    positions = np.array([gmap.keyframes[kf_id].t.flatten() for kf_id in sorted_kf_ids])
    x_coords = positions[:, 0]
    z_coords = positions[:, 2]  # Typically Z is the forward direction, Y is down

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, z_coords, marker='o', markersize=4, linestyle='-', color='royalblue', label='Camera Path')
    plt.scatter(x_coords[0], z_coords[0], c='lime', s=100, label='Start', zorder=5, edgecolors='black')
    plt.scatter(x_coords[-1], z_coords[-1], c='red', s=100, label='Current', zorder=5, edgecolors='black')

    plt.title(f"Camera Trajectory (Top-Down View) - Keyframe {sorted_kf_ids[-1]}")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal') 

    plot_filename = os.path.join(output_dir, f"trajectory_kf_{sorted_kf_ids[-1]:04d}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"    -> Saved 2D trajectory plot to {plot_filename}")

def plot_and_save_trajectory_3d(gmap: Map, output_dir: str):
    """Generates a 3D plot of the camera's trajectory and poses."""
    os.makedirs(output_dir, exist_ok=True)
    sorted_kf_ids = sorted(gmap.keyframes.keys())
    if not sorted_kf_ids:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = []
    orientations = []

    for kf_id in sorted_kf_ids:
        kf = gmap.keyframes[kf_id]
        positions.append(kf.t.flatten())
        orientations.append(kf.R[:, 2].flatten()) # Z-axis of camera's local frame

    positions = np.array(positions)
    orientations = np.array(orientations)

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='grey', linestyle='--', label='Path')
    ax.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        orientations[:, 0], orientations[:, 1], orientations[:, 2],
        length=0.5, normalize=True, color='blue', label='Camera Pose'
    )
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='lime', s=100, label='Start', edgecolors='black')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, label='Current', edgecolors='black')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'3D Camera Trajectory - Keyframe {sorted_kf_ids[-1]}')
    ax.legend()
    
    # Auto-scaling for consistent aspect ratio
    max_range = np.array([positions[:, i].max() - positions[:, i].min() for i in range(3)]).max() / 2.0
    mid = np.mean(positions, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plot_filename = os.path.join(output_dir, f"trajectory_3d_kf_{sorted_kf_ids[-1]:04d}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"    -> Saved 3D trajectory plot to {plot_filename}")