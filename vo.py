import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg') # Use a non-GUI backend to prevent Qt errors
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
        return self.orb.detectAndCompute(image, None)

class FeatureMatcher:
    """Abstract base class for feature matchers."""
    def match(self, des1, des2):
        raise NotImplementedError

class BruteForceMatcher(FeatureMatcher):
    """Concrete Brute-Force feature matcher with ratio test."""
    def __init__(self, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    def match(self, des1, des2):
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
    The main class to orchestrate the Visual Odometry pipeline.
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
        Processes a single frame to estimate pose and generate visualizations.
        Returns visualization images for keypoints and matches.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Feature Extraction
        keypoints, descriptors = self.feature_extractor.extract(gray_frame)
        keypoints_img = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

        if self.previous_frame_data is None:
            print("Initializing first frame.")
            self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
            # Return blank image for matches on the first frame
            return keypoints_img, np.zeros((frame.shape[0], frame.shape[1] * 2, 3), dtype=np.uint8)

        # Handle cases with no descriptors
        if descriptors is None or self.previous_frame_data['descriptors'] is None:
            print("Warning: No descriptors found, skipping frame.")
            self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
            return keypoints_img, np.zeros((frame.shape[0], frame.shape[1] * 2, 3), dtype=np.uint8)

        # 2. Feature Matching
        matches = self.feature_matcher.match(self.previous_frame_data['descriptors'], descriptors)
        matches_img = cv2.drawMatches(
            self.previous_frame_data['image'], self.previous_frame_data['keypoints'],
            gray_frame, keypoints, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 3. Pose Estimation
        if len(matches) > 15:
            relative_R, relative_t = self._estimate_pose(matches, self.previous_frame_data['keypoints'], keypoints)
            
            if relative_R is not None and relative_t is not None:
                # 4. Trajectory Integration
                self._integrate_pose(relative_R, relative_t)

        self.previous_frame_data = {'image': gray_frame, 'keypoints': keypoints, 'descriptors': descriptors}
        
        return keypoints_img, matches_img

    def _estimate_pose(self, matches, kp1, kp2):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None
            
        _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        return R_rel, t_rel

    def _integrate_pose(self, R_rel, t_rel):
        # NOTE: Using a fixed scale factor as true scale is unknown from monocular images.
        scale = 1.0
        self.global_t = self.global_t + self.global_R @ (t_rel * scale)
        self.global_R = self.global_R @ R_rel
        self.trajectory.append(self.global_t.flatten())

def main():
    # --- Configuration ---
    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi'  # <--- CHANGE THIS to your video file path
    
    # Camera Intrinsics (from your camera_info message)
    camera_matrix = np.array([
        [431.39865, 0.0, 429.08605],
        [0.0, 431.39865, 235.27142],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties for output writers
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    keypoints_writer = cv2.VideoWriter('keypoints_video.mp4', fourcc, fps, (frame_width, frame_height))
    # Matches video is wider because it shows two images side-by-side
    matches_writer = cv2.VideoWriter('matches_video.mp4', fourcc, fps, (frame_width * 2, frame_height))

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
        
        #print(f"Processing frame {frame_count}...")
        keypoints_img, matches_img = vo_pipeline.process_frame(frame)

        # Write frames to output videos
        keypoints_writer.write(keypoints_img)
        
        # Resize matches image if it's not the correct size (can happen on first frame)
        if matches_img.shape[1] != frame_width * 2:
             matches_img = cv2.resize(matches_img, (frame_width * 2, frame_height))
        matches_writer.write(matches_img)
        
        frame_count += 1

    print("Video processing finished.")
    
    # --- Cleanup and Final Visualization ---
    cap.release()
    keypoints_writer.release()
    matches_writer.release()
    cv2.destroyAllWindows()
    
    # Plot the final trajectory
    trajectory_points = np.array(vo_pipeline.trajectory)
    plt.figure(figsize=(8, 8))
    # Plotting X and Z for a top-down view
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 2], marker='o', linestyle='-', label='Camera Path')
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.title('Estimated Camera Trajectory (Top-Down View)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig('trajectory.png')
    # plt.show() # This line is removed to prevent GUI-related errors

if __name__ == '__main__':
    main()



#    video_path = '/home/students/girgine/ros2_ws/src/visual_odometry/data/srge_lab.avi'  # <--- CHANGE THIS to your video file path
