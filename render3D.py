"""
Visualize Cached NeRF Data with PROPER COLOR CALCULATION
Uses both UVW and Beta for view-dependent colors
"""

import numpy as np
import open3d as o3d

def load_all_cached_data(scene_name='fern', N=512, M=200, sigma_thresh=0):
    """Load ALL cached NeRF data including beta"""
    prefix = f'fastnerf/output/{scene_name}'
    
    print(f"Loading cached data for {scene_name}...")
    print(f"  - Index map (voxel occupancy)")
    inds = np.load(f'{prefix}_inds_{N}_{sigma_thresh}.npy')
    
    print(f"  - UVWS (base colors + density)")
    uvws = np.load(f'{prefix}_uvws_{N}_{sigma_thresh}.npy')
    
    print(f"  - Beta (directional appearance)")
    beta = np.load(f'{prefix}_beta_{M}_cart.npy')
    
    print(f"\n✓ Index map shape: {inds.shape}")
    print(f"✓ UVWS shape: {uvws.shape} -> [uvw_x, uvw_y, uvw_z, sigma, ...]")
    print(f"✓ Beta shape: {beta.shape} -> directional components")
    print(f"✓ Number of occupied voxels: {uvws.shape[0]:,}")
    
    return inds, uvws, beta, N, M

def compute_proper_colors(uvw, beta_grid, view_dir, M, latent_channels=8):
    """
    Compute proper NeRF colors using both UVW and Beta
    
    CORRECT FORMULA from nerf.py:
        uvw shape: (3, latent_channels) - per point
        beta shape: (latent_channels,) - per direction
        rgb = sum(uvw * beta, axis=-1) = (3,) 
    
    Args:
        uvw: (N, 3*latent_channels) or (N, 3, latent_channels) base color components
        beta_grid: (M, M, M, latent_channels) directional appearance grid
        view_dir: (3,) normalized viewing direction
        M: Beta grid resolution
        latent_channels: number of latent code channels (default 8)
    
    Returns:
        colors: (N, 3) RGB colors
    """
    # Normalize view direction
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-6)
    
    # Map direction to beta grid indices
    # Beta grid spans [-1, 1] in each dimension
    beta_idx = ((view_dir + 1) / 2 * M).astype(int)
    beta_idx = np.clip(beta_idx, 0, M - 1)
    
    # Look up beta value for this viewing direction
    beta_value = beta_grid[beta_idx[0], beta_idx[1], beta_idx[2]]  # (latent_channels,)
    
    # Reshape UVW to (N, 3, latent_channels) if needed
    N = uvw.shape[0]
    if uvw.shape[1] != 3:
        # If uvw is flattened (N, 3*latent_channels), reshape it
        uvw_reshaped = uvw[:, :3*latent_channels].reshape(N, 3, latent_channels)
    else:
        uvw_reshaped = uvw
    
    # CORRECT FORMULA: RGB = sum(UVW * Beta, axis=-1)
    # This is element-wise multiplication followed by sum over latent channels
    colors = (uvw_reshaped * beta_value[np.newaxis, np.newaxis, :]).sum(axis=-1)  # (N, 3)
    
    # Clamp to [0, 1] as done in the model
    colors = np.clip(colors, 0, 1)
    
    return colors

def extract_point_cloud_with_proper_colors(inds, uvws, beta, N, M, 
                                           density_threshold=-0.5, 
                                           max_points=200000,
                                           view_dir=None):
    """
    Extract point cloud with PROPER color calculation
    """
    print(f"\nExtracting points with density > {density_threshold}...")
    
    # Default viewing direction (looking at scene from front)
    if view_dir is None:
        view_dir = np.array([0.0, 0.0, -1.0])  # Looking down -Z axis
    
    # Get coordinates of occupied voxels
    coords = np.argwhere(inds >= 0)
    valid_inds = inds[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Extract UVW and sigma
    # UVW is stored as (3 * latent_channels) = 24 channels for latent_channels=8
    # Sigma is the last channel
    latent_channels = 8  # Default in fastnerf
    uvw = uvws[valid_inds, :3*latent_channels]  # First 24 channels (3*8)
    sigma = uvws[valid_inds, 3*latent_channels]  # Channel 24 is sigma
    
    print(f"Density range: [{sigma.min():.2f}, {sigma.max():.2f}]")
    suggested_threshold = np.percentile(sigma, 75)
    print(f"Suggested threshold (75th percentile): {suggested_threshold:.2f}")
    
    # Filter by density
    mask = sigma > density_threshold
    coords_filtered = coords[mask]
    uvw_filtered = uvw[mask]
    sigma_filtered = sigma[mask]
    
    print(f"Points above threshold: {coords_filtered.shape[0]:,}")
    
    if coords_filtered.shape[0] == 0:
        print(f"⚠ No points! Try lower threshold (current: {density_threshold})")
        return None
    
    # Downsample if needed
    if coords_filtered.shape[0] > max_points:
        print(f"Downsampling to {max_points:,} points...")
        indices = np.random.choice(coords_filtered.shape[0], max_points, replace=False)
        coords_filtered = coords_filtered[indices]
        uvw_filtered = uvw_filtered[indices]
        sigma_filtered = sigma_filtered[indices]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Normalize coordinates to [-1, 1]
    points = coords_filtered.astype(np.float32)
    points = (points / N) * 2 - 1
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"UVW shape: {uvw_filtered.shape}, Beta shape: {beta.shape}")
    
    colors = compute_proper_colors(uvw_filtered, beta, view_dir, M, latent_channels=8)
    
    print(f"Color range after computation: [{colors.min():.2f}, {colors.max():.2f}]")
    
    # Colors are already in [0, 1] from the clamp in compute_proper_colorssigmoid to handle large ranges
    # colors = 1 / (1 + np.exp(-colors))
    
    colors = np.clip(colors, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"✓ Created point cloud with {len(pcd.points):,} points")
    print(f"✓ Colors computed using UVW + Beta(view_dir)")
    
    return pcd

def visualize_multiple_viewpoints(inds, uvws, beta, N, M, 
                                  density_threshold=-0.5,
                                  max_points=100000):
    """
    Show the same scene from different viewpoints with different colors
    This demonstrates the view-dependent appearance!
    """
    print("\n" + "="*70)
    print("View-Dependent Color Demonstration")
    print("="*70)
    
    # Different viewing directions
    viewpoints = {
        'Front (0,0,-1)': np.array([0, 0, -1]),
        'Top (0,-1,0)': np.array([0, -1, 0]),
        'Right (1,0,0)': np.array([1, 0, 0]),
        'Angle (1,1,-1)': np.array([1, 1, -1]),
    }
    
    print("\nGenerating point clouds from different viewpoints...")
    print("(Colors should change based on viewing direction!)\n")
    
    pcds = []
    for name, view_dir in viewpoints.items():
        print(f"\n--- {name} ---")
        pcd = extract_point_cloud_with_proper_colors(
            inds, uvws, beta, N, M,
            density_threshold=density_threshold,
            max_points=max_points,
            view_dir=view_dir
        )
        if pcd is not None:
            pcds.append((name, pcd))
    
    # Visualize all together
    print("\n" + "="*70)
    print("Showing all viewpoints (colors represent different viewing angles)")
    print("="*70)
    
    geometries = []
    offset = 2.5  # Spacing between point clouds
    for i, (name, pcd) in enumerate(pcds):
        # Translate each point cloud so they're side by side
        translated_pcd = pcd.translate([i * offset, 0, 0])
        geometries.append(translated_pcd)
        
        # Add text label (coordinate frame)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_frame.translate([i * offset, -1.5, 0])
        geometries.append(coord_frame)
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="View-Dependent Colors (Left to Right: Front, Top, Right, Angle)",
        width=1920,
        height=1080
    )

def visualize_single_view(pcd, scene_name='fern'):
    """Visualize single point cloud"""
    print("\n" + "="*70)
    print("Open3D Interactive Viewer")
    print("="*70)
    print("\nControls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Q or ESC: Quit")
    print("\n" + "="*70 + "\n")
    
    # Estimate normals
    try:
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    except:
        pass
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"NeRF Cached Data - {scene_name} (Proper Colors)", 
                      width=1280, height=720)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    opt.show_coordinate_frame = True
    
    vis.run()
    vis.destroy_window()

def main():
    """Main function"""
    print("="*70)
    print("Cached NeRF Data Visualizer (WITH PROPER COLORS)")
    print("="*70)
    print("\nThis version uses BOTH UVW and Beta for correct colors!")
    print("Colors will be view-dependent (like the real NeRF model).\n")
    
    # Configuration
    scene_name = 'horns'
    N = 512
    M = 200
    sigma_thresh = 0
    density_threshold = -2
    max_points = 200000
    
    try:
        # Load ALL data including beta
        inds, uvws, beta, N, M = load_all_cached_data(scene_name, N, M, sigma_thresh)
        
        print("\n" + "="*70)
        print("Choose visualization mode:")
        print("="*70)
        print("1. Single view (default front view)")
        print("2. Multiple viewpoints comparison (shows view-dependent colors)")
        print("="*70)
        
        choice = input("\nChoice (1 or 2, default=1): ").strip() or "1"
        
        if choice == "2":
            # Show multiple viewpoints
            visualize_multiple_viewpoints(
                inds, uvws, beta, N, M,
                density_threshold=density_threshold,
                max_points=100000  # Fewer points for multiple views
            )
        else:
            # Single view
            view_dir = np.array([1, 1, -1])  # Front view
            pcd = extract_point_cloud_with_proper_colors(
                inds, uvws, beta, N, M,
                density_threshold=density_threshold,
                max_points=max_points,
                view_dir=view_dir
            )
            
            if pcd is not None:
                visualize_single_view(pcd, scene_name)
        
    except ImportError:
        print("\n❌ Open3D not installed!")
        print("Install with: pip install open3d")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find cached files!")
        print(f"   {e}")
        print("\nMake sure these files exist:")
        print(f"  - fastnerf/output/{scene_name}_inds_{N}_{sigma_thresh}.npy")
        print(f"  - fastnerf/output/{scene_name}_uvws_{N}_{sigma_thresh}.npy")
        print(f"  - fastnerf/output/{scene_name}_beta_{M}_cart.npy")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
