import open3d as o3d
import numpy as np
import os
import matplotlib.cm as cm

class PointCloudVisualizer:
    """
    A class to visualize a sequence of point cloud files from a directory.
    """
    def __init__(self, pcd_folder):
        """
        Initializes the visualizer.

        Args:
            pcd_folder (str): The path to the folder containing .pcd files.
        """
        if not os.path.isdir(pcd_folder):
            print(f"Error: Directory not found at '{pcd_folder}'")
            exit()

        # Get and sort all .pcd files
        self.pcd_files = sorted([
            os.path.join(pcd_folder, f)
            for f in os.listdir(pcd_folder)
            if f.endswith('.pcd')
        ])

        if not self.pcd_files:
            print(f"Error: No .pcd files found in '{pcd_folder}'")
            exit()

        # Visualization state
        self.current_index = -1
        self.pcd_geometry = o3d.geometry.PointCloud()

        # Setup the visualizer window and callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Point Cloud Viewer | Press 'N' for Next, 'Q' to Quit", 1280, 720)
        self.setup_renderer()
        self.register_callbacks()

    def setup_renderer(self):
        """Configures the rendering options."""
        opt = self.vis.get_render_option()
        opt.point_size = 10.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark background

    def register_callbacks(self):
        """Registers key press callbacks."""
        # Register 'N' key to advance to the next frame
        self.vis.register_key_callback(ord("N"), self.advance_frame)

    def advance_frame(self, vis):
        """
        Callback function to load and display the next point cloud.
        This function is triggered by pressing the 'N' key.
        """
        self.current_index += 1
        if self.current_index >= len(self.pcd_files):
            print("Reached the end of the sequence. Looping back to the start.")
            self.current_index = 0

        self.update_visualization()
        return True

    def update_visualization(self):
        """
        Loads a point cloud, colors it, and updates the visualizer.
        """
        filepath = self.pcd_files[self.current_index]
        filename = os.path.basename(filepath)
        print(f"Displaying: {filename}")

        # Load the new point cloud
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_points():
            print(f"Warning: {filename} is empty. Skipping.")
            # Copy an empty point cloud to clear the view
            self.pcd_geometry.points = o3d.utility.Vector3dVector()
            self.pcd_geometry.colors = o3d.utility.Vector3dVector()
        else:
            # --- Color point cloud based on depth (Z-axis) ---
            points = np.asarray(pcd.points)
            depths = points[:, 2] # Z-coordinate represents depth

            # Normalize depths to the [0, 1] range for color mapping
            min_depth, max_depth = np.min(depths), np.max(depths)
            if max_depth - min_depth > 1e-6:
                norm_depths = (depths - min_depth) / (max_depth - min_depth)
            else:
                norm_depths = np.zeros_like(depths) # Handle case of flat point cloud

            # Use a colormap (e.g., viridis) to get RGB values
            colors = cm.viridis(norm_depths)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Update the geometry in the scene
            self.pcd_geometry.points = pcd.points
            self.pcd_geometry.colors = pcd.colors

        # Update the geometry and reset the camera view
        self.vis.update_geometry(self.pcd_geometry)
        self.vis.update_renderer()
        # --- FIX ---
        # The method reset_camera_bounding_box() was removed in newer Open3D versions.
        # reset_view_point(True) is the correct way to reset the view to the geometry bounds.
        self.vis.reset_view_point(True)
        
    def run(self):
        """
        Starts the visualization loop.
        """
        # --- Add static geometry that does not change ---
        # 1. Coordinate Axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(axes, reset_bounding_box=False)

        # 2. Ground Plane
        ground_plane = o3d.geometry.TriangleMesh.create_box(width=20, height=0.01, depth=20)
        ground_plane.compute_vertex_normals()
        ground_plane.paint_uniform_color([0.3, 0.3, 0.3]) # Dark gray
        # Center the plane and place its top surface at Y=0
        ground_plane.translate([-10, -10, -10])
        self.vis.add_geometry(ground_plane, reset_bounding_box=False)

        # Add the point cloud geometry object (initially empty)
        self.vis.add_geometry(self.pcd_geometry, reset_bounding_box=False)
        
        # Load the very first frame to start
        self.advance_frame(self.vis)

        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()
        print("Visualizer closed.")

def main():
    """Main function to run the visualizer."""
    # IMPORTANT: This should be the folder where your .pcd files are saved.
    pcd_folder = 'point_clouds_pcd'
    
    visualizer = PointCloudVisualizer(pcd_folder)
    visualizer.run()

if __name__ == '__main__':
    main()
