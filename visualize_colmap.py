"""
Visualize COLMAP reconstruction to check quality before NeRF training

USAGE:
python visualize_colmap.py --sparse_dir /path/to/sparse/0

This will show:
1. Camera poses in 3D
2. 3D point cloud
3. Camera coverage statistics
4. Potential issues
"""

import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse


def load_reconstruction(sparse_path):
    """Load COLMAP reconstruction"""
    reconstruction = pycolmap.Reconstruction(str(sparse_path))
    
    print("="*60)
    print("COLMAP Reconstruction Summary")
    print("="*60)
    print(f"✓ Images: {len(reconstruction.images)}")
    print(f"✓ 3D points: {len(reconstruction.points3D)}")
    print(f"✓ Cameras: {len(reconstruction.cameras)}")
    
    return reconstruction


def check_reconstruction_quality(reconstruction):
    """Check for common issues"""
    
    print("\n" + "="*60)
    print("Quality Checks")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check 1: Number of images
    num_images = len(reconstruction.images)
    if num_images < 10:
        issues.append(f"Too few images ({num_images}). Recommend 20+ for forward-facing, 40+ for 360°")
    elif num_images < 20:
        warnings.append(f"Low image count ({num_images}). More images = better quality")
    else:
        print(f"✓ Image count: {num_images} (Good)")
    
    # Check 2: Number of 3D points
    num_points = len(reconstruction.points3D)
    if num_points < 1000:
        issues.append(f"Too few 3D points ({num_points}). Scene might lack texture")
    elif num_points < 5000:
        warnings.append(f"Low 3D point count ({num_points}). Add more texture to scene")
    else:
        print(f"✓ 3D points: {num_points} (Good)")
    
    # Check 3: Points per image
    total_observations = sum(len(img.points2D) for img in reconstruction.images.values())
    avg_points_per_image = total_observations / num_images if num_images > 0 else 0
    
    if avg_points_per_image < 100:
        issues.append(f"Too few points per image ({avg_points_per_image:.0f}). Images might be blurry or lack overlap")
    elif avg_points_per_image < 500:
        warnings.append(f"Low points per image ({avg_points_per_image:.0f})")
    else:
        print(f"✓ Avg points per image: {avg_points_per_image:.0f} (Good)")
    
    # Check 4: Camera poses distribution
    camera_centers = []
    for image in reconstruction.images.values():
        cam_from_world = image.cam_from_world()
        # Camera center in world coordinates
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        center = -R.T @ t
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    
    # Check if cameras are too clustered
    distances = []
    for i in range(len(camera_centers)-1):
        dist = np.linalg.norm(camera_centers[i+1] - camera_centers[i])
        distances.append(dist)
    
    if len(distances) > 0:
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if std_dist / (avg_dist + 1e-6) > 2.0:
            warnings.append(f"Irregular camera spacing (std/mean = {std_dist/avg_dist:.2f})")
        else:
            print(f"✓ Camera spacing: consistent")
    
    # Check 5: Scene scale
    points_3d = np.array([p.xyz for p in reconstruction.points3D.values()])
    scene_extent = np.ptp(points_3d, axis=0)
    scene_size = np.linalg.norm(scene_extent)
    
    print(f"✓ Scene extent: [{scene_extent[0]:.2f}, {scene_extent[1]:.2f}, {scene_extent[2]:.2f}]")
    print(f"✓ Scene size: {scene_size:.2f} units")
    
    # Check 6: Point cloud density
    camera_center = np.mean(camera_centers, axis=0)
    point_distances = np.linalg.norm(points_3d - camera_center, axis=1)
    
    near = np.percentile(point_distances, 1)
    far = np.percentile(point_distances, 99)
    depth_range = far / near if near > 0 else 0
    
    print(f"✓ Depth range: {near:.2f} to {far:.2f} (ratio: {depth_range:.1f}x)")
    
    if depth_range > 100:
        warnings.append(f"Very large depth range ({depth_range:.1f}x). Might affect NeRF quality")
    
    # Print issues and warnings
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed! Reconstruction looks good.")
    
    return len(issues) == 0


def visualize_reconstruction(reconstruction, save_path=None):
    """Visualize camera poses and 3D points"""
    
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    
    # Extract camera positions and orientations
    camera_positions = []
    camera_directions = []
    
    for image in reconstruction.images.values():
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        
        # Camera center in world coordinates
        center = -R.T @ t
        camera_positions.append(center)
        
        # Camera forward direction (negative z-axis in camera frame)
        forward = -R.T @ np.array([0, 0, 1])
        camera_directions.append(forward)
    
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    
    # Extract 3D points (subsample for visualization)
    points_3d = np.array([p.xyz for p in reconstruction.points3D.values()])
    
    # Subsample points if too many
    if len(points_3d) > 10000:
        indices = np.random.choice(len(points_3d), 10000, replace=False)
        points_3d = points_3d[indices]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot 3D points
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                c='lightgray', s=1, alpha=0.3, label='3D Points')
    
    # Plot camera positions
    ax1.scatter(camera_positions[:, 0], 
                camera_positions[:, 1], 
                camera_positions[:, 2],
                c='red', s=50, label='Cameras')
    
    # Plot camera directions
    scale = 0.2
    for pos, direction in zip(camera_positions, camera_directions):
        ax1.quiver(pos[0], pos[1], pos[2],
                   direction[0], direction[1], direction[2],
                   color='blue', alpha=0.6, length=scale, arrow_length_ratio=0.3)
    
    # Connect cameras in sequence
    if len(camera_positions) > 1:
        ax1.plot(camera_positions[:, 0], 
                 camera_positions[:, 1], 
                 camera_positions[:, 2],
                 'g--', alpha=0.5, linewidth=1, label='Camera Path')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('COLMAP Reconstruction (3D View)')
    ax1.legend()
    
    # Set equal aspect ratio
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                          points_3d[:, 1].max() - points_3d[:, 1].min(),
                          points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], 
                c='lightgray', s=1, alpha=0.3, label='3D Points')
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1],
                c='red', s=50, label='Cameras', zorder=10)
    
    # Plot camera viewing directions (top-down)
    scale = 0.3
    for pos, direction in zip(camera_positions, camera_directions):
        ax2.arrow(pos[0], pos[1], 
                  direction[0] * scale, direction[1] * scale,
                  head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
    
    # Connect cameras
    if len(camera_positions) > 1:
        ax2.plot(camera_positions[:, 0], camera_positions[:, 1],
                 'g--', alpha=0.5, linewidth=1, label='Camera Path')
    
    # Number the cameras
    for i, pos in enumerate(camera_positions[::max(1, len(camera_positions)//20)]):
        ax2.annotate(str(i), (pos[0], pos[1]), fontsize=8, color='darkred')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-Down View')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()
    print("✓ Visualization complete")


def check_poses_bounds(poses_bounds_path):
    """Check poses_bounds.npy quality"""
    
    print("\n" + "="*60)
    print("Checking poses_bounds.npy")
    print("="*60)
    
    if not Path(poses_bounds_path).exists():
        print(f"❌ File not found: {poses_bounds_path}")
        return False
    
    poses_bounds = np.load(poses_bounds_path)
    
    print(f"✓ Shape: {poses_bounds.shape}")
    print(f"✓ Expected: (N, 17) where N = number of images")
    
    if poses_bounds.shape[1] != 17:
        print(f"❌ Wrong shape! Should be (N, 17), got {poses_bounds.shape}")
        return False
    
    # Extract info
    n_images = poses_bounds.shape[0]
    
    # Each row: [R(9) | t(3) | hwf(3) | near(1) | far(1)]
    Rs = poses_bounds[:, :9].reshape(-1, 3, 3)
    ts = poses_bounds[:, 9:12]
    hwfs = poses_bounds[:, 12:15]
    nears = poses_bounds[:, 15]
    fars = poses_bounds[:, 16]
    
    print(f"\n✓ Images: {n_images}")
    print(f"✓ Image size: {int(hwfs[0, 0])}x{int(hwfs[0, 1])}")
    print(f"✓ Focal length: {hwfs[0, 2]:.1f}")
    print(f"✓ Near bound: {nears.min():.3f} - {nears.max():.3f}")
    print(f"✓ Far bound: {fars.min():.3f} - {fars.max():.3f}")
    
    # Check rotation matrices are valid
    issues = []
    for i, R in enumerate(Rs):
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.1:
            issues.append(f"Image {i}: Invalid rotation matrix (det={det:.3f})")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more")
        return False
    else:
        print("\n✅ poses_bounds.npy looks valid!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize COLMAP reconstruction quality")
    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="Path to sparse/0 directory")
    parser.add_argument("--poses_bounds", type=str, default=None,
                        help="Optional: Path to poses_bounds.npy to check")
    parser.add_argument("--save_viz", type=str, default=None,
                        help="Save visualization to this path")
    
    args = parser.parse_args()
    
    sparse_path = Path(args.sparse_dir)
    
    if not sparse_path.exists():
        print(f"❌ Sparse directory not found: {sparse_path}")
        return
    
    # Load reconstruction
    reconstruction = load_reconstruction(sparse_path)
    
    # Check quality
    is_good = check_reconstruction_quality(reconstruction)
    
    # Visualize
    save_path = args.save_viz if args.save_viz else sparse_path.parent.parent / "reconstruction_viz.png"
    visualize_reconstruction(reconstruction, save_path)
    
    # Check poses_bounds if provided
    if args.poses_bounds:
        check_poses_bounds(args.poses_bounds)
    else:
        # Try to find poses_bounds.npy automatically
        poses_bounds_path = sparse_path.parent.parent / "poses_bounds.npy"
        if poses_bounds_path.exists():
            check_poses_bounds(poses_bounds_path)
    
    # Final recommendation
    print("\n" + "="*60)
    if is_good:
        print("✅ RECOMMENDATION: Reconstruction looks good! Safe to train NeRF.")
    else:
        print("⚠ RECOMMENDATION: Fix issues above before training NeRF.")
    print("="*60)


if __name__ == "__main__":
    main()