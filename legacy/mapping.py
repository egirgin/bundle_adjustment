import cv2
import numpy as np
import os
import open3d as o3d

class FeatureExtractor:
    """Abstract base class for feature extractors."""
    def extract(self, image):
        raise NotImplementedError

class ORBExtractor(FeatureExtractor):
    """Concrete ORB feature extractor."""
    def __init__(self, n_features=3000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image):
        # Extracts keypoints and descriptors from an image.
        return self.orb.detectAndCompute(image, None)

class FeatureMatcher:
    """Abstract base class for feature matchers."""
    def match(self, des1, des2):
        raise NotImplementedError

class BruteForceMatcher(FeatureMatcher):
    """Concrete Brute-Force feature matcher with Lowe's ratio test."""
    def __init__(self, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    def match(self, des1, des2):
        # Finds the best matches between two sets of descriptors.
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass # Ignore cases where knnMatch returns fewer than 2 matches
        return good_matches

class VisualOdometryPipeline:
    """
    The main class to orchestrate Visual Odometry and Point Cloud registration.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.previous_frame_data = None

        # NEW: Initialize global pose (world rotation and translation)
        # The global pose starts as the identity matrix (no rotation)
        # and a zero vector (at the origin).
        self.world_R = np.eye(3)
        self.world_t = np.zeros((3, 1))

    def process_frame(self, frame):
        """
        Processes a single frame to estimate pose and returns triangulated points
        in the global coordinate system.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        
        # On the first frame, just store its data and exit
        if self.previous_frame_data is None:
            print("Initializing first frame...")
            self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
            return None

        # Match features between the previous and current frames
        matches = self.feature_matcher.match(self.previous_frame_data['descriptors'], descriptors)

        # We need a minimum number of matches to reliably estimate pose
        if len(matches) > 15:
            # Estimate the *relative* pose between the last two frames
            relative_R, relative_t, pts1, pts2 = self._estimate_pose(matches, self.previous_frame_data['keypoints'], keypoints)
            
            if relative_R is not None and relative_t is not None:
                # Triangulate points to get their 3D position *relative* to the previous frame
                points_3d_relative = self._triangulate_points(relative_R, relative_t, pts1, pts2)
                
                if points_3d_relative is not None:
                    # NEW: Transform the relative 3D points into the global coordinate system
                    # by applying the current global rotation and translation.
                    points_3d_world = (self.world_R @ points_3d_relative) + self.world_t

                    # NEW: Update the global pose by composing the new relative transformation.
                    # The new global position is the old position plus the rotated relative translation.
                    self.world_t = self.world_t + self.world_R @ relative_t
                    # The new global rotation is the composition of the old rotation and the new relative rotation.
                    self.world_R = self.world_R @ relative_R

                    # Update the previous frame data for the next iteration
                    self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
                    return points_3d_world

        # If not enough matches, just update the frame data without returning points
        self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
        return None

    def _estimate_pose(self, matches, kp1, kp2):
        # Get the coordinates of the matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find the Essential matrix, which describes the camera's relative motion
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None
            
        # Recover the relative rotation (R_rel) and translation (t_rel) from the Essential matrix
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        # Filter out outlier points using the mask from recoverPose
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]

        return R_rel, t_rel, inlier_pts1, inlier_pts2

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        # We need points to triangulate
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            return None
        
        # Create projection matrices for triangulation
        # The first camera is at the origin of its own coordinate system
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # The second camera's projection matrix is defined by its relative rotation and translation
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        # Triangulate points to get 4D homogeneous coordinates
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D coordinates by dividing by the 4th component
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6) # Add epsilon for stability
        
        # Filter out points that are behind the camera (have a non-positive Z value)
        valid_mask = points_3d[2, :] > 0
        points_3d = points_3d[:, valid_mask]

        if points_3d.shape[1] == 0:
            return None

        return points_3d[0:3, :]

def main():
    """Main function to run the Visual Odometry and Point Cloud Registration pipeline."""
    # --- Configuration ---
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi' # <--- CHANGE THIS
    output_dir = 'output_map' # MODIFIED: Directory for the final map
    os.makedirs(output_dir, exist_ok=True)
    final_map_path = os.path.join(output_dir, "final_map.pcd")

    # --- Camera Intrinsics ---
    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # Assuming no lens distortion

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, ORBExtractor(), BruteForceMatcher())
    
    # NEW: Create a global point cloud object to accumulate points
    global_pcd = o3d.geometry.PointCloud()

    # --- Main Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_count}...")
        # MODIFIED: Process frame and get points in the world coordinate system
        new_points_world = vo_pipeline.process_frame(frame)
        
        # NEW: If new points were generated, add them to the global map
        if new_points_world is not None:
            print(f"  -> Adding {new_points_world.shape[1]} new points to the map.")
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(new_points_world.T)
            global_pcd += temp_pcd

        frame_count += 1

    print("\nVideo processing finished.")
    
    # --- Finalization and Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    
    # NEW: Save the final, accumulated point cloud map
    if global_pcd.has_points():
        print(f"\nSaving final registered point cloud map to '{final_map_path}'...")
        # Voxel downsampling cleans up the point cloud by merging close points,
        # reducing noise and the total number of points. 0.05 means a 5cm grid.
        downsampled_pcd = global_pcd.voxel_down_sample(voxel_size=0.1)
        o3d.io.write_point_cloud(final_map_path, downsampled_pcd)
        print("Map saved successfully.")

        # --- Visualization ---
        print("\nDisplaying the final map. Press 'q' in the window to close.")
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("\nNo points were generated for the final map.")


if __name__ == '__main__':
    main()