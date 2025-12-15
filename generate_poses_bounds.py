import pycolmap
import numpy as np
from pathlib import Path

# Paths
sparse_path = Path(r"C:\Users\WELTINFO\Desktop\models\NotVeryFastNeRF\data\myData\chjra\sparse\0")

# Load reconstruction
reconstruction = pycolmap.Reconstruction(str(sparse_path))

# Calculate bounds
points = np.array([p.xyz for p in reconstruction.points3D.values()])
center = np.mean(points, axis=0)
distances = np.linalg.norm(points - center, axis=1)
near = max(0.01, np.percentile(distances, 1) * 0.8)
far = np.percentile(distances, 99) * 1.2

print(f"Depth bounds: near={near:.3f}, far={far:.3f}")

poses_bounds_list = []

for image_id in sorted(reconstruction.images.keys()):
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    
    # Get the transform (note the parentheses!)
    cam_from_world = image.cam_from_world()
    
    # World-to-camera -> Camera-to-world
    R_w2c = cam_from_world.rotation.matrix()
    t_w2c = cam_from_world.translation
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    
    # COLMAP uses OpenCV convention: right, down, forward (X, Y, Z)
    # LLFF expects: down, right, back
    # So we need to convert: [right, down, forward] -> [down, right, back]
    # This is: [X, Y, Z] -> [Y, X, -Z]
    # Reorder columns: new_R[:, 0] = old_R[:, 1] (down)
    #                  new_R[:, 1] = old_R[:, 0] (right)
    #                  new_R[:, 2] = -old_R[:, 2] (back = -forward)
    R_llff = np.zeros_like(R_c2w)
    R_llff[:, 0] = R_c2w[:, 1]   # down (Y)
    R_llff[:, 1] = R_c2w[:, 0]   # right (X)
    R_llff[:, 2] = -R_c2w[:, 2]  # back (-Z)
    
    # Camera params
    h = camera.height
    w = camera.width
    focal = camera.params[0]
    
    # Build pose matrix (3x5) with LLFF convention
    pose = np.zeros((3, 5))
    pose[:, :3] = R_llff
    pose[:, 3] = t_c2w
    pose[0, 4] = h
    pose[1, 4] = w
    pose[2, 4] = focal
    
    # Add bounds
    pose_with_bounds = np.concatenate([pose.flatten(), [near, far]])
    poses_bounds_list.append(pose_with_bounds)

poses_bounds = np.array(poses_bounds_list)

# Save
output_file = Path(r"C:\Users\WELTINFO\Desktop\models\NotVeryFastNeRF\data\myData\chjra\poses_bounds.npy")
np.save(output_file, poses_bounds)

print(f"\n✅ Saved: {output_file}")
print(f"Shape: {poses_bounds.shape}")
print(f"Number of poses: {len(poses_bounds_list)}")
print(f"\nCamera info:")
print(f"  Resolution: {int(w)}x{int(h)}")
print(f"  Focal length: {focal:.1f} pixels")
