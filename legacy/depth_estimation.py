import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-GUI backend
import matplotlib.pyplot as plt

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
        # Finds the best matches between two descriptor sets.
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        try:
            # Apply ratio test to find good matches
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            # Handle cases where not enough matches are found
            pass
        return good_matches

class VisualOdometryPipeline:
    """
    The main class to orchestrate the Visual Odometry and Depth Estimation pipeline.
    """
    def __init__(self, camera_matrix, dist_coeffs, feature_extractor, feature_matcher):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        
        # State variables
        self.previous_frame_data = None
        self.global_R = np.eye(3)
        self.global_t = np.zeros((3, 1))
        self.trajectory = [self.global_t.flatten()]

    def process_frame(self, frame):
        """
        Processes a single frame to estimate pose, triangulate points, and generate visualizations.
        Returns visualization images for keypoints, matches, and depth.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Feature Extraction
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        keypoints_img = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

        # Create blank images for outputs in case of initialization or failure
        h, w, _ = frame.shape
        matches_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        depth_map_img = np.zeros_like(frame)

        if self.previous_frame_data is None:
            print("Initializing first frame...")
            self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
            return keypoints_img, matches_img, depth_map_img

        # 2. Feature Matching
        matches = self.feature_matcher.match(self.previous_frame_data['descriptors'], descriptors)
        matches_img = cv2.drawMatches(
            self.previous_frame_data['image'], self.previous_frame_data['keypoints'],
            gray_frame, keypoints, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 3. Pose Estimation & Triangulation
        if len(matches) > 15: # Need enough matches for robust estimation
            # Estimate relative pose between frames
            relative_R, relative_t, pts1, pts2 = self._estimate_pose(matches, self.previous_frame_data['keypoints'], keypoints)
            
            if relative_R is not None and relative_t is not None:
                # 4. Trajectory Integration
                self._integrate_pose(relative_R, relative_t)

                # 5. Triangulation and Depth Visualization
                depth_map_img = self._triangulate_and_visualize_depth(relative_R, relative_t, pts1, pts2, frame)

        # Update the previous frame data for the next iteration
        self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
        
        return keypoints_img, matches_img, depth_map_img

    def _estimate_pose(self, matches, kp1, kp2):
        """Estimates the relative rotation and translation between two frames."""
        # Get the coordinates of the matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find the Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None, None
            
        # Recover the relative Rotation and Translation
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        # Get inlier points
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]

        return R_rel, t_rel, inlier_pts1, inlier_pts2

    def _integrate_pose(self, R_rel, t_rel):
        """Updates the global pose of the camera."""
        # NOTE: Using a fixed scale factor as true scale is unknown from monocular images.
        scale = 1.0
        self.global_t = self.global_t + self.global_R @ (t_rel * scale)
        self.global_R = self.global_R @ R_rel
        self.trajectory.append(self.global_t.flatten())

    def _triangulate_and_visualize_depth(self, R_rel, t_rel, pts1, pts2, frame):
        """Triangulates 3D points and creates a depth map visualization on the grayscale frame."""
        # --- MODIFICATION ---
        # Start with the grayscale image instead of a black canvas
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to BGR to be able to draw colored circles on it
        depth_map_vis = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Robustness check: Ensure there are points to triangulate
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            return depth_map_vis
        
        # Create projection matrices for triangulation
        P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.camera_matrix @ np.hstack((R_rel, t_rel))

        # Transpose points for triangulation function which expects (2, N) format
        pts1_transposed = pts1.T
        pts2_transposed = pts2.T

        # Triangulate points to get 3D coordinates
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_transposed, pts2_transposed)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d_hom / (points_4d_hom[3] + 1e-6) # Add epsilon to avoid division by zero
        
        # Filter out points that are behind the camera (z <= 0)
        valid_mask = points_3d[2, :] > 0
        points_3d = points_3d[:, valid_mask]
        valid_pts2 = pts2[valid_mask]

        if points_3d.shape[1] == 0:
            return depth_map_vis # Return blank map if no valid points

        # Get depth values (Z coordinate)
        depths = points_3d[2, :]
        
        # Normalize depths for visualization
        # Using percentile to avoid extreme outliers skewing the color map
        min_depth = np.percentile(depths, 5)
        max_depth = np.percentile(depths, 95)
        
        # Avoid division by zero if all depths are the same
        if max_depth - min_depth < 1e-6:
            normalized_depths = np.ones_like(depths) * 255
        else:
            normalized_depths = 255 * (depths - min_depth) / (max_depth - min_depth)
            normalized_depths = np.clip(normalized_depths, 0, 255)

        # Create a colored depth visualization
        for i, pt2d in enumerate(valid_pts2):
            x, y = int(pt2d[0]), int(pt2d[1])
            depth_val = normalized_depths[i]
            # Use a colormap (JET) to represent depth
            color = cv2.applyColorMap(np.uint8([[depth_val]]), cv2.COLORMAP_JET)[0][0]
            cv2.circle(depth_map_vis, (x, y), 7, color.tolist(), -1)

        return depth_map_vis

def main():
    """Main function to run the VO pipeline and generate output videos."""
    # --- Configuration ---
    # IMPORTANT: Change this to your video file path
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi'  # <--- CHANGE THIS to your video file path
    
    # Camera Intrinsics (replace with your camera's calibration)
    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    # Assuming no lens distortion
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        print("Please make sure the video file is in the same directory or provide the full path.")
        return

    # Get video properties for output writers
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    keypoints_writer = cv2.VideoWriter('keypoints_video.mp4', fourcc, fps, (frame_width, frame_height))
    matches_writer = cv2.VideoWriter('matches_video.mp4', fourcc, fps, (frame_width * 2, frame_height))
    depth_writer = cv2.VideoWriter('depth_video.mp4', fourcc, fps, (frame_width, frame_height)) # <-- New video writer for depth

    # Instantiate the pipeline with swappable components
    orb_extractor = ORBExtractor()
    bf_matcher = BruteForceMatcher()
    vo_pipeline = VisualOdometryPipeline(camera_matrix, dist_coeffs, orb_extractor, bf_matcher)

    # --- Main Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_count}...")
        keypoints_img, matches_img, depth_img = vo_pipeline.process_frame(frame)

        # Write frames to output videos
        keypoints_writer.write(keypoints_img)
        depth_writer.write(depth_img)
        
        # Resize matches image if it's not the correct size (can happen on first frame)
        if matches_img.shape[1] != frame_width * 2 or matches_img.shape[0] != frame_height:
             matches_img = cv2.resize(matches_img, (frame_width * 2, frame_height))
        matches_writer.write(matches_img)
        
        frame_count += 1

    print("\nVideo processing finished.")
    print("Output files created: keypoints_video.mp4, matches_video.mp4, depth_video.mp4, trajectory.png")
    
    # --- Cleanup and Final Visualization ---
    cap.release()
    keypoints_writer.release()
    matches_writer.release()
    depth_writer.release() # Release the new writer
    cv2.destroyAllWindows()
    
    # Plot the final trajectory
    trajectory_points = np.array(vo_pipeline.trajectory)
    plt.figure(figsize=(8, 8))
    # Plotting X and Z for a top-down view of the camera path
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 2], marker='.', linestyle='-', label='Camera Path')
    plt.xlabel('X position (meters)')
    plt.ylabel('Z position (meters)')
    plt.title('Estimated Camera Trajectory (Top-Down View)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig('trajectory.png')

if __name__ == '__main__':
    main()
