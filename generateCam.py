"""
Generate poses_bounds.npy from images using OpenCV Structure-from-Motion
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob


def load_images(images_dir):
    """Load all images from directory"""
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.*')))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) < 10:
        print(f"WARNING: Only {len(image_paths)} images found. Recommend 20-30 for best results.")
    
    print(f"Found {len(image_paths)} images")
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load {path}")
    
    return images, image_paths


def calibrate_camera_with_checkerboard(calibration_dir, pattern_size=(9, 6)):
    """Calibrate camera using checkerboard pattern images
    
    Args:
        calibration_dir: Directory containing checkerboard calibration images
        pattern_size: Tuple (width, height) of internal corners in checkerboard
                     Default (9,6) is for standard 10x7 checkerboard
    
    Returns:
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        image_size: (width, height)
        
    Note: Checkerboard should be printed and photographed from various angles.
          Capture 15-20 images for best results.
    """
    
    print("\n" + "="*60)
    print("CALIBRATING CAMERA WITH CHECKERBOARD")
    print("="*60)
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Load calibration images
    cal_images = sorted(glob.glob(os.path.join(calibration_dir, '*.*')))
    cal_images = [p for p in cal_images if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(cal_images) < 10:
        print(f"WARNING: Only {len(cal_images)} calibration images. Recommend 15-20 for accuracy.")
    
    img_shape = None
    successful_detections = 0
    
    for img_path in cal_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            successful_detections += 1
            print(f"✓ Detected pattern in {os.path.basename(img_path)}")
        else:
            print(f"✗ Pattern not found in {os.path.basename(img_path)}")
    
    if successful_detections < 10:
        raise ValueError(f"Only {successful_detections} patterns detected. Need at least 10 for reliable calibration.")
    
    print(f"\nCalibrating with {successful_detections} images...")
    
    # Calibrate camera
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        raise ValueError("Camera calibration failed!")
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_reproj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"\n✓ Calibration successful!")
    print(f"✓ Reprojection error: {mean_error:.4f} pixels (lower is better)")
    print(f"✓ Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"✓ Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    
    if mean_error > 1.0:
        print(f"WARNING: Reprojection error is high. Consider recapturing calibration images.")
    
    return K, dist_coeffs, img_shape


def estimate_camera_intrinsics(image):
    """Estimate camera intrinsics using simple heuristic (fallback method)
    
    Note: This is a rough approximation. For accurate results, use 
    calibrate_camera_with_checkerboard() with proper calibration images.
    """
    
    h, w = image.shape[:2]
    
    # Estimate focal length as 1.2 * max(width, height)
    # Rationale: Most cameras have FOV around 50-70 degrees
    # focal ≈ image_size / (2 * tan(FOV/2)) ≈ 1.2 * max(w,h) for ~60° FOV
    focal = 1.2 * max(w, h)
    
    # Principal point at image center (standard assumption)
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K, h, w, focal


def extract_features(images):
    """Extract SIFT features from all images
    
    SIFT (Scale-Invariant Feature Transform) detects distinctive keypoints
    that are robust to scale, rotation, and illumination changes.
    """
    
    print("\n" + "="*60)
    print("STEP 1: Feature Detection")
    print("="*60)
    
    # Use SIFT detector (2000 features per image is a good balance)
    sift = cv2.SIFT_create(nfeatures=2000)
    
    all_keypoints = []
    all_descriptors = []
    
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        print(f"Image {i+1}: Found {len(kp)} keypoints")
    
    return all_keypoints, all_descriptors


def match_features(descriptors1, descriptors2):
    """Match features between two images using FLANN matcher
    
    Uses Lowe's ratio test to filter ambiguous matches:
    A match is good if nearest neighbor is significantly closer than second-nearest
    """
    
    # FLANN (Fast Library for Approximate Nearest Neighbors) - faster than BFMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test: reject if nearest/second-nearest ratio > 0.7
    # This filters out ambiguous matches where multiple features look similar
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    return good_matches


def estimate_pose_pairwise(kp1, kp2, matches, K):
    """Estimate relative pose between two images using Essential matrix
    
    Process:
    1. Find Essential matrix using RANSAC (robust to outliers)
    2. Decompose E into rotation (R) and translation (t)
    3. Choose physically valid solution (positive depth)
    
    Requires: At least 8 point correspondences (Essential matrix has 5 DOF)
    """
    
    if len(matches) < 8:
        return None, None, None
    
    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find Essential matrix using RANSAC
    # E encodes epipolar geometry: p2^T @ E @ p1 = 0
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None:
        return None, None, None
    
    # Recover pose from Essential matrix (4 possible solutions, picks correct one)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    # Filter inliers (matches consistent with the pose)
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    
    return R, t, inlier_matches


def triangulate_points(kp1, kp2, matches, K, R1, t1, R2, t2):
    """Triangulate 3D points from matched features
    
    Given two camera poses and corresponding 2D points, compute 3D point positions.
    Uses linear triangulation (DLT - Direct Linear Transform).
    """
    
    # Projection matrices: P = K[R|t] maps 3D world points to 2D image
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    
    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T
    
    # Triangulate using DLT (solves for X in: p1 = P1*X and p2 = P2*X)
    points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points3D = points4D[:3] / points4D[3]  # Convert from homogeneous to 3D
    
    return points3D.T


def reconstruct_scene(images, all_keypoints, all_descriptors, K):
    """Incremental Structure-from-Motion reconstruction"""
    
    print("\n" + "="*60)
    print("STEP 2: Pairwise Matching")
    print("="*60)
    
    n_images = len(images)
    
    # Initialize with first two images
    matches_01 = match_features(all_descriptors[0], all_descriptors[1])
    print(f"Images 0-1: {len(matches_01)} matches")
    
    R, t, inlier_matches = estimate_pose_pairwise(
        all_keypoints[0], all_keypoints[1], matches_01, K
    )
    
    if R is None:
        print("ERROR: Failed to estimate initial pose. Try with different images.")
        sys.exit(1)
    
    # Camera poses (world-to-camera)
    # First camera at origin
    poses = []
    poses.append({
        'R': np.eye(3),
        't': np.zeros((3, 1))
    })
    
    # Second camera pose
    poses.append({
        'R': R,
        't': t
    })
    
    print("\n" + "="*60)
    print("STEP 3: Incremental Reconstruction")
    print("="*60)
    
    # Triangulate initial 3D points
    points3D = triangulate_points(
        all_keypoints[0], all_keypoints[1], inlier_matches,
        K, poses[0]['R'], poses[0]['t'], poses[1]['R'], poses[1]['t']
    )
    
    print(f"Initial reconstruction: {len(points3D)} 3D points")
    
    # Add remaining images incrementally
    for i in range(2, n_images):
        # Match with previous image
        matches = match_features(all_descriptors[i-1], all_descriptors[i])
        print(f"Images {i-1}-{i}: {len(matches)} matches")
        
        R, t, inlier_matches = estimate_pose_pairwise(
            all_keypoints[i-1], all_keypoints[i], matches, K
        )
        
        if R is None:
            print(f"Warning: Failed to estimate pose for image {i}, using last pose")
            # Fallback: reuse previous pose (not ideal but prevents crash)
            R = poses[-1]['R'].copy()
            t = poses[-1]['t'].copy()
        else:
            # IMPORTANT: Chain the poses to get absolute world coordinates
            # R, t are relative: they transform from camera[i-1] to camera[i]
            # We need absolute poses in world frame
            
            R_prev = poses[-1]['R']  # Previous camera's world-to-camera rotation
            t_prev = poses[-1]['t']  # Previous camera's position
            
            # Compose transformations (world -> cam[i-1] -> cam[i]):
            # If prev is [R_prev|t_prev] and relative is [R|t], then:
            # New absolute pose is: R_new = R @ R_prev, t_new = R @ t_prev + t
            R_new = R @ R_prev
            t_new = R @ t_prev + t
            
            R = R_new
            t = t_new
        
        poses.append({'R': R, 't': t})
        
        # Triangulate additional points
        if len(inlier_matches) >= 8:
            new_points = triangulate_points(
                all_keypoints[i-1], all_keypoints[i], inlier_matches,
                K, poses[i-1]['R'], poses[i-1]['t'], poses[i]['R'], poses[i]['t']
            )
            points3D = np.vstack([points3D, new_points])
    
    print(f"\nFinal reconstruction: {len(points3D)} 3D points")
    
    return poses, points3D


def convert_poses_to_c2w(poses):
    """Convert world-to-camera poses to camera-to-world"""
    
    c2w_poses = []
    for pose in poses:
        R = pose['R']
        t = pose['t']
        
        # Convert world-to-camera to camera-to-world transformation
        # If [R|t] transforms world points to camera: p_cam = R * p_world + t
        # Then camera-to-world is: p_world = R^T * (p_cam - t) = R^T * p_cam - R^T * t
        # So: c2w = [R^T | -R^T @ t]
        R_c2w = R.T
        t_c2w = -R.T @ t
        
        c2w_poses.append({
            'R': R_c2w,
            't': t_c2w
        })
    
    return c2w_poses


def calculate_depth_bounds(poses, points3D):
    """Calculate near and far depth bounds from the entire scene
    
    Note: Using global bounds (same for all cameras) is more stable than per-camera bounds.
    This ensures consistent depth range across all views.
    """
    
    # Calculate scene center and spread
    center = np.mean(points3D, axis=0)
    distances_from_center = np.linalg.norm(points3D - center, axis=1)
    
    # Use percentiles to exclude outliers
    # Near: conservative (1st percentile * 0.8 margin)
    # Far: generous (99th percentile * 1.2 margin)
    near = max(0.01, np.percentile(distances_from_center, 1) * 0.8)
    far = np.percentile(distances_from_center, 99) * 1.2
    
    print(f"\nDepth bounds (global): near={near:.3f}, far={far:.3f}")
    
    # Return same bounds for all images (more stable for NeRF training)
    bounds_per_image = [[near, far] for _ in range(len(poses))]
    
    return np.array(bounds_per_image)


def create_poses_bounds(poses, K, h, w, focal, points3D):
    """Create poses_bounds.npy array in LLFF format
    
    LLFF (Local Light Field Fusion) expects poses in a specific coordinate convention:
    - Standard OpenCV/COLMAP: X=right, Y=down, Z=forward (into scene)
    - LLFF convention: X=down, Y=right, Z=backward (away from scene)
    
    This requires reordering the rotation matrix columns.
    """
    
    print("\n" + "="*60)
    print("STEP 4: Creating poses_bounds.npy")
    print("="*60)
    
    # Calculate depth bounds
    bounds = calculate_depth_bounds(poses, points3D)
    
    poses_bounds_data = []
    
    for i, pose in enumerate(poses):
        R_c2w = pose['R']  # Camera-to-world rotation (OpenCV convention)
        t_c2w = pose['t'].reshape(3, 1)  # Camera position in world
        
        # CRITICAL: Convert OpenCV convention to LLFF convention
        # OpenCV: [right, down, forward] = [X, Y, Z]
        # LLFF:   [down, right, back]    = [Y, X, -Z]
        #
        # Reorder columns of rotation matrix:
        R_llff = np.zeros_like(R_c2w)
        R_llff[:, 0] = R_c2w[:, 1]   # LLFF X (down) = OpenCV Y
        R_llff[:, 1] = R_c2w[:, 0]   # LLFF Y (right) = OpenCV X  
        R_llff[:, 2] = -R_c2w[:, 2]  # LLFF Z (back) = -OpenCV Z (negated forward)
        
        # Build pose matrix (3x5): [R_llff | t | [H, W, f]]
        # Note: translation vector stays the same (it's a position, not a direction)
        pose_matrix = np.zeros((3, 5))
        pose_matrix[:, :3] = R_llff  # Use LLFF-convention rotation
        pose_matrix[:, 3] = t_c2w.flatten()  # Camera position (unchanged)
        pose_matrix[0, 4] = h  # Image height
        pose_matrix[1, 4] = w  # Image width
        pose_matrix[2, 4] = focal  # Focal length in pixels
        
        # Flatten and add bounds
        # Final format: 17 values = 15 (pose matrix) + 2 (near, far)
        pose_flat = pose_matrix.flatten()  # 15 values
        pose_with_bounds = np.concatenate([pose_flat, bounds[i]])  # 17 values
        
        poses_bounds_data.append(pose_with_bounds)
    
    poses_bounds = np.array(poses_bounds_data)
    
    print(f"✓ Created poses_bounds array")
    print(f"✓ Shape: {poses_bounds.shape}")
    print(f"✓ Number of images: {len(poses)}")
    print(f"✓ Depth range: {bounds[:, 0].min():.2f} to {bounds[:, 1].max():.2f}")
    
    return poses_bounds


def main():
    parser = argparse.ArgumentParser(description="Generate poses_bounds.npy from images using OpenCV SfM")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for poses_bounds.npy (default: parent of images_dir)")
    parser.add_argument("--calibration_dir", type=str, default=None,
                        help="Directory with checkerboard calibration images (optional, for accurate intrinsics)")
    parser.add_argument("--checkerboard_size", type=str, default="9x6",
                        help="Checkerboard pattern size as WxH (default: 9x6 for 10x7 board)")
    
    args = parser.parse_args()
    
    images_dir = os.path.abspath(args.images_dir)
    
    if args.output_dir is None:
        output_dir = os.path.dirname(images_dir)
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    # Validate inputs
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("OpenCV Structure-from-Motion Pipeline")
    print("="*60)
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load images
    images, image_paths = load_images(images_dir)
    
    if len(images) < 3:
        print("ERROR: Need at least 3 images for reconstruction")
        sys.exit(1)
    
    # Get camera intrinsics (calibrated or estimated)
    if args.calibration_dir and os.path.exists(args.calibration_dir):
        # Parse checkerboard size
        try:
            cb_w, cb_h = map(int, args.checkerboard_size.split('x'))
            pattern_size = (cb_w, cb_h)
        except:
            print(f"ERROR: Invalid checkerboard size format: {args.checkerboard_size}")
            print("Use format: WxH (e.g., 9x6)")
            sys.exit(1)
        
        # Calibrate using checkerboard
        try:
            K, dist_coeffs, (w, h) = calibrate_camera_with_checkerboard(
                args.calibration_dir, pattern_size
            )
            focal = (K[0, 0] + K[1, 1]) / 2.0  # Average focal length
            
            # Optionally undistort scene images if significant distortion
            if np.abs(dist_coeffs).max() > 0.1:
                print(f"\nUndistorting images (distortion detected)...")
                h_int, w_int = images[0].shape[:2]
                newK, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w_int, h_int), 1, (w_int, h_int))
                images = [cv2.undistort(img, K, dist_coeffs, None, newK) for img in images]
                K = newK
                print("✓ Images undistorted")
        except Exception as e:
            print(f"ERROR: Calibration failed: {e}")
            print("Falling back to heuristic estimation...")
            K, h, w, focal = estimate_camera_intrinsics(images[0])
    else:
        # Use heuristic estimation
        if args.calibration_dir:
            print(f"WARNING: Calibration directory not found: {args.calibration_dir}")
            print("Using heuristic estimation instead.\n")
        
        K, h, w, focal = estimate_camera_intrinsics(images[0])
    
    print(f"\nCamera parameters:")
    print(f"  Image size: {int(w)} x {int(h)}")
    print(f"  Focal length: {focal:.1f} pixels")
    
    # Extract features
    all_keypoints, all_descriptors = extract_features(images)
    
    # Reconstruct scene
    poses, points3D = reconstruct_scene(images, all_keypoints, all_descriptors, K)
    
    # Convert to camera-to-world poses
    c2w_poses = convert_poses_to_c2w(poses)
    
    # Create poses_bounds array
    poses_bounds = create_poses_bounds(c2w_poses, K, h, w, focal, points3D)
    
    # Save to file
    output_file = os.path.join(output_dir, "poses_bounds.npy")
    np.save(output_file, poses_bounds)
    
    print("\n" + "="*60)
    print("SUCCESS! Your dataset is ready.")
    print("="*60)
    print(f"✓ Saved: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Verify poses with visualization tool")
    print(f"  2. Train NeRF model on this dataset")
    print(f"\nNote: For best quality, consider using COLMAP instead of this OpenCV approach.")


if __name__ == "__main__":
    # Usage examples:
    # 
    # Basic usage (heuristic intrinsics):
    #   python generateCam.py --images_dir data/myData/scene_name/images
    # 
    # With custom output directory:
    #   python generateCam.py --images_dir data/myData/scene_name/images --output_dir data/myData/scene_name
    # 
    # With camera calibration (recommended for accuracy):
    #   python generateCam.py --images_dir data/myData/scene_name/images --calibration_dir calibration_images
    # 
    # With custom checkerboard pattern:
    #   python generateCam.py --images_dir data/myData/scene_name/images --calibration_dir calibration_images --checkerboard_size 7x5
    main()
