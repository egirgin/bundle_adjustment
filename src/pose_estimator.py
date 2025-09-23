# vo_project/pose_estimator.py

import cv2
import numpy as np
from typing import List, Tuple, Optional

def estimate_pose(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    camera_matrix: np.ndarray
) -> Tuple[
    Optional[np.ndarray],  # R_rel
    Optional[np.ndarray],  # t_rel
    Optional[np.ndarray],  # inlier pts1
    Optional[np.ndarray],  # inlier pts2
    Optional[np.ndarray]   # inlier indices
]:
    """
    Estimates the relative pose between two frames.
    """
    pts1: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    E: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    if E is None:
        return None, None, None, None, None
        
    _, R_rel, t_rel, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)

    num_matches: int = len(matches)
    num_inliers: int = np.sum(mask)
    inlier_ratio: float = num_inliers / num_matches if num_matches > 0 else 0
    print(f"    -> Pose Estimation: {num_inliers} inliers out of {num_matches} matches (Ratio: {inlier_ratio:.2f})")

    # A very low ratio indicates a poor geometric match
    if inlier_ratio < 0.4: # You can tune this threshold
        print("    -> WARNING: Low inlier ratio. Pose estimate may be unreliable.")

    inlier_mask: np.ndarray = mask.ravel() == 1
    return R_rel, t_rel, pts1[inlier_mask], pts2[inlier_mask], np.where(inlier_mask)[0]


def estimate_pose_pnp(
    points_3d: np.ndarray, 
    points_2d: np.ndarray, 
    camera_matrix: np.ndarray, 
    dist_coeffs: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimates the camera pose from 3D-2D correspondences using PnP with RANSAC.

    Args:
        points_3d (np.ndarray): Array of 3D points in the world coordinate system.
        points_2d (np.ndarray): Array of corresponding 2D points in the image plane.
        camera_matrix (np.ndarray): The intrinsic camera matrix.
        dist_coeffs (np.ndarray, optional): The distortion coefficients. Defaults to None.

    Returns:
        tuple: A tuple containing the rotation vector and translation vector, or (None, None) if pose estimation fails.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    if len(points_3d) < 4:
        print("    -> PnP Skipped: Not enough points for PnP.")
        return None, None

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            print(f"    -> PnP Pose Estimation: {len(inliers)} inliers out of {len(points_2d)} correspondences.")
            return rvec, tvec
        else:
            print("    -> PnP RANSAC failed to find a valid pose.")
            return None, None

    except cv2.error as e:
        print(f"    -> An error occurred during solvePnPRansac: {e}")
        return None, None