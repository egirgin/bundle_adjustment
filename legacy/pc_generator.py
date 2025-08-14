import cv2
import numpy as np
import os
import open3d as o3d  # NEW: Import the Open3D library

# REMOVED: The old manual .ply saving function is no longer needed.

# NEW: Helper function to save point clouds to a .pcd file using Open3D
def save_point_cloud_o3d(filepath, points_3d):
    """
    Saves a 3D point cloud to a .pcd file using Open3D.
    Expects points_3d to be a (3, N) numpy array.
    """
    # Check if there are any points to save
    if points_3d is None or points_3d.shape[1] == 0:
        print("Warning: No points to save.")
        return

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Open3D expects points as a Vector3dVector, which can be created from a numpy array of shape (N, 3)
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)

    # Save the point cloud to a .pcd file
    o3d.io.write_point_cloud(filepath, pcd)


class FeatureExtractor:
    """Abstract base class for feature extractors."""
    def extract(self, image):
        raise NotImplementedError

class ORBExtractor(FeatureExtractor):
    """Concrete ORB feature extractor."""
    def __init__(self, n_features=3000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image):
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
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass
        return good_matches

class VisualOdometryPipeline:
    """
    The main class to orchestrate the Visual Odometry and Point Cloud Generation pipeline.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.previous_frame_data = None

    # MODIFIED: The process_frame method now calls the new Open3D save function
    def process_frame(self, frame, frame_count, output_dir):
        """
        Processes a single frame to estimate pose, triangulate points, and save a point cloud.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)

        if self.previous_frame_data is None:
            print("Initializing first frame...")
            self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}

        matches = self.feature_matcher.match(self.previous_frame_data['descriptors'], descriptors)

        if len(matches) > 15:
            relative_R, relative_t, pts1, pts2 = self._estimate_pose(matches, self.previous_frame_data['keypoints'], keypoints)
            
            if relative_R is not None and relative_t is not None:
                points_3d = self._triangulate_points(relative_R, relative_t, pts1, pts2)
                
                if points_3d is not None:
                    # MODIFIED: Filename now uses .pcd extension
                    filepath = os.path.join(output_dir, f"frame_{frame_count:05d}.pcd")
                    # MODIFIED: Call the new save function
                    save_point_cloud_o3d(filepath, points_3d)
                    print(f"Saved point cloud with {points_3d.shape[1]} points to {filepath}")

        self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
        

    def _estimate_pose(self, matches, kp1, kp2):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]

        return R_rel, t_rel, inlier_pts1, inlier_pts2

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            return None
        
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))
        
        pts1_transposed = pts1.T
        pts2_transposed = pts2.T
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_transposed, pts2_transposed)
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6) 
        
        valid_mask = points_3d[2, :] > 0
        points_3d = points_3d[:, valid_mask]

        if points_3d.shape[1] == 0:
            return None

        return points_3d[0:3, :]

def main():
    """Main function to run the Point Cloud Generation pipeline."""
    # --- Configuration ---
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi' # <--- CHANGE THIS
    output_dir = 'point_clouds_pcd' # MODIFIED: New directory for .pcd files
    os.makedirs(output_dir, exist_ok=True)

    # --- Camera Intrinsics ---
    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, ORBExtractor(), BruteForceMatcher())

    # --- Main Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_count}...")
        vo_pipeline.process_frame(frame, frame_count, output_dir)
        frame_count += 1

    print("\nVideo processing finished.")
    print(f"Output files created in ./{output_dir}/ and keypoints_video.mp4")
    
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()