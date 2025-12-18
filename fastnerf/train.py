"""
Neural Radiance Fields (NeRF) Training Script
==============================================

This script implements training for FastNeRF, which is an optimized version of the original NeRF.
NeRF learns a continuous 5D function (3D position + 2D viewing direction) that maps to:
- RGB color (3 values)
- Volume density (1 value, sigma)

The model can then synthesize novel views of the scene by querying this learned function.
"""

import os, sys
from opt import get_opts  # Hyperparameters and command-line arguments
import torch
from collections import defaultdict
import csv
from datetime import datetime

from torch.utils.data import DataLoader
from datasets import dataset_dict  # LLFF and Blender dataset loaders

# models
from models.nerf import Embedding, NeRF  # Positional encoding and MLP network
from models.rendering import render_rays  # Volume rendering equation implementation

# optimizer, scheduler, visualization
from utils import *  # Training utilities (optimizers, schedulers, learning rate helpers)

# losses
from losses import loss_dict  # MSE loss for RGB reconstruction

# metrics
from metrics import *  # PSNR (Peak Signal-to-Noise Ratio) for evaluation

# pytorch-lightning: High-level training framework
from lightning.pytorch.callbacks import ModelCheckpoint  # Saves best models during training
from lightning import LightningModule, Trainer  # Base class and training orchestrator
from lightning.pytorch.loggers import TensorBoardLogger  # Logs metrics for visualization

class NeRFSystem(LightningModule):
    """
    Main training system for NeRF using PyTorch Lightning.
    
    This class encapsulates the entire NeRF training pipeline including:
    1. Model architecture (coarse and fine networks)
    2. Data loading and preprocessing
    3. Training and validation loops
    4. Loss computation and optimization
    """
    
    def __init__(self, hparams):
        """
        Initialize the NeRF training system.
        
        Args:
            hparams: Hyperparameters from opt.py containing:
                - Dataset settings (root_dir, img_wh, dataset_name)
                - Sampling settings (N_samples, N_importance)
                - Training settings (batch_size, lr, num_epochs)
                - Model settings (loss_type, optimizer)
        """
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)  # Saves hparams to checkpoint for reproducibility

        # Initialize loss function (default: MSE between predicted and ground truth RGB)
        self.loss = loss_dict[hparams.loss_type]()

        # Storage for validation outputs across batches
        self.validation_steps_outputs = []
        
        # Create CSV file for logging detailed training metrics
        self.loss_log_dir = os.path.join('logs', hparams.exp_name)
        os.makedirs(self.loss_log_dir, exist_ok=True)
        self.loss_log_file = os.path.join(self.loss_log_dir, 'training_losses.csv')
        
        # Initialize CSV file with headers
        with open(self.loss_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'global_step', 'train_loss', 'train_psnr', 'learning_rate', 'timestamp'])

        # ===== POSITIONAL ENCODING =====
        # Transforms low-dimensional inputs to high-dimensional space for better learning
        # Using Fourier features: [x, sin(2^0*x), cos(2^0*x), ..., sin(2^9*x), cos(2^9*x)]
        # This helps the network learn high-frequency details
        
        # Position encoding: 3D coordinates -> 63D (3 + 3*10*2)
        self.embedding_xyz = Embedding(3, 10)  # 10 frequency bands
        
        # Direction encoding: 3D viewing direction -> 27D (3 + 3*4*2)
        self.embedding_dir = Embedding(3, 4)   # 4 frequency bands (fewer for view-dependent effects)
        
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        # ===== NEURAL NETWORK MODELS =====
        # Coarse network: Learns basic scene geometry with uniform sampling
        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        
        # Fine network: Refines details using importance sampling (optional but recommended)
        # Only created if N_importance > 0 (hierarchical volume sampling)
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        """
        Extract rays and ground truth RGB values from a batch.
        
        Args:
            batch: Dictionary containing:
                - 'rays': (B, 8) [origin_xyz(3), direction_xyz(3), near(1), far(1)]
                - 'rgbs': (B, 3) Ground truth RGB colors for each ray
        
        Returns:
            rays: (B, 8) Ray parameters
            rgbs: (B, 3) Target RGB colors
        """
        rays = batch['rays']  # (B, 8) Ray origin, direction, and depth bounds
        rgbs = batch['rgbs']  # (B, 3) Ground truth RGB values for supervision
        return rays, rgbs

    def forward(self, rays):
        """
        Perform volume rendering for a batch of rays.
        
        This is the core inference function that:
        1. Samples points along each ray
        2. Queries the NeRF network at sampled positions
        3. Applies volume rendering to produce final RGB and depth
        
        The rendering is done in chunks to avoid GPU out-of-memory errors.
        
        Args:
            rays: (B, 8) Tensor containing ray information:
                - rays[:, 0:3]: Ray origins (camera positions)
                - rays[:, 3:6]: Ray directions (normalized)
                - rays[:, 6:7]: Near bound (minimum depth)
                - rays[:, 7:8]: Far bound (maximum depth)
        
        Returns:
            results: Dictionary with rendered outputs:
                - 'rgb_coarse': (B, 3) RGB from coarse network
                - 'depth_coarse': (B,) Depth map from coarse network
                - 'rgb_fine': (B, 3) RGB from fine network (if N_importance > 0)
                - 'depth_fine': (B,) Depth map from fine network (if N_importance > 0)
        """
        B = rays.shape[0]  # Batch size (number of rays)
        results = defaultdict(list)
        
        # Process rays in chunks to manage GPU memory
        # Typical chunk size: 32K rays at a time
        for i in range(0, B, self.hparams.chunk):
            # Render a chunk of rays using volume rendering
            rendered_ray_chunks = \
                render_rays(self.models,              # [coarse_model, fine_model]
                            self.embeddings,           # [xyz_embedding, dir_embedding]
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,    # Number of coarse samples (e.g., 64)
                            self.hparams.use_disp,     # Sample in disparity space?
                            self.hparams.perturb,      # Add noise to sample positions
                            self.hparams.noise_std,    # Noise for regularization
                            self.hparams.N_importance, # Number of fine samples (e.g., 128)
                            self.hparams.chunk,        # Chunk size for network queries
                            self.train_dataset.white_back)  # White or black background

            # Collect results from each chunk
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        # Concatenate all chunks back together
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        """
        Prepare training and validation datasets.
        
        Datasets precompute rays from camera parameters:
        1. Load images and camera poses (intrinsics + extrinsics)
        2. Generate rays for each pixel in each image
        3. Each ray stores: origin, direction, near/far bounds, and ground truth RGB
        
        Two dataset types supported:
        - 'llff': Real-world forward-facing scenes (8 views for validation)
        - 'blender': Synthetic 360-degree scenes with known camera poses
        """
        dataset = dataset_dict[self.hparams.dataset_name]  # Get dataset class
        kwargs = {
            'root_dir': self.hparams.root_dir,  # Path to dataset
            'img_wh': tuple(self.hparams.img_wh)  # Image resolution [width, height]
        }
        
        # LLFF-specific settings for real-world captures
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses  # Are cameras on sphere?
            kwargs['val_num'] = self.hparams.num_gpus  # Number of validation images
            
        # Create train/val splits
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        """
        Set up optimizer and learning rate scheduler.
        
        Optimizer options (from opt.py):
        - Adam (default): Adaptive learning rate, good for NeRF
        - RAdam: Rectified Adam with warm-up
        - Ranger: RAdam + Lookahead
        - SGD: Basic stochastic gradient descent
        
        Scheduler options:
        - StepLR (default): Decay LR at specific epochs
        - Cosine: Cosine annealing schedule
        - Poly: Polynomial decay
        
        Returns:
            List of optimizers and list of schedulers
        """
        # Create optimizer for all model parameters (coarse + fine networks)
        self.optimizer = get_optimizer(self.hparams, self.models)
        
        # Create learning rate scheduler (reduces LR over time for better convergence)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        """
        Create training data loader.
        
        During training, we sample random rays from random images.
        Typical batch_size: 1024 rays (not full images).
        This allows the network to see diverse views quickly.
        
        Returns:
            DataLoader that yields batches of rays
        """
        return DataLoader(self.train_dataset,
                          shuffle=True,        # Shuffle rays for better training
                          num_workers=4,       # Parallel data loading
                          batch_size=self.hparams.batch_size,  # Rays per batch (e.g., 1024)
                          pin_memory=True)     # Speed up GPU transfer

    def val_dataloader(self):
        """
        Create validation data loader.
        
        During validation, we render entire images (all H*W rays at once).
        This allows us to compute image-level metrics (PSNR, SSIM) and visualize results.
        
        Returns:
            DataLoader that yields full images as ray batches
        """
        return DataLoader(self.val_dataset,
                          shuffle=False,       # Don't shuffle for consistent validation
                          num_workers=4,
                          batch_size=1,        # Validate one full image at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        """
        Execute one training step (forward pass + loss computation).
        
        This is called for each batch during training. The complete flow is:
        
        1. DECODE BATCH: Extract rays and ground truth RGB values
        2. FORWARD PASS: 
           a. Sample points along rays (uniform + importance sampling)
           b. Query NeRF network(s) at each point -> RGB + density (sigma)
           c. Volume rendering: Integrate along ray using quadrature
              RGB(r) = Σ T_i * alpha_i * c_i
              where:
              - T_i = exp(-Σ sigma_j * delta_j) is transmittance (accumulated opacity)
              - alpha_i = 1 - exp(-sigma_i * delta_i) is local opacity
              - c_i is the RGB color at point i
              - delta_i is the distance between samples
        
        3. LOSS COMPUTATION:
           - Compare predicted RGB with ground truth RGB using MSE
           - If using fine network, loss = MSE(coarse) + MSE(fine)
        
        4. METRICS: Compute PSNR for monitoring (higher = better)
           PSNR = -10 * log10(MSE), typically 20-35 dB for good results
        
        Args:
            batch: Dictionary with 'rays' and 'rgbs'
            batch_nb: Batch number (not used)
        
        Returns:
            Dictionary with 'loss' for backpropagation
        """
        # Track current learning rate for logging
        log = {'lr': get_learning_rate(self.optimizer)}
        
        # Extract rays and ground truth colors
        rays, rgbs = self.decode_batch(batch)
        
        # FORWARD PASS: Render rays through the NeRF
        # This is where the magic happens!
        # - Samples 64 points uniformly along each ray (coarse sampling)
        # - Queries NeRF network to get RGB + density at each point
        # - Applies volume rendering equation to get final pixel color
        # - If fine network exists, samples 128 more points at important regions
        results = self(rays)
        
        # LOSS CALCULATION
        # Compare predicted RGB with ground truth using Mean Squared Error
        # If fine network exists: loss = MSE(coarse_rgb, gt) + MSE(fine_rgb, gt)
        log['train/loss'] = loss = self.loss(results, rgbs)
        
        # Determine which network's output to use for metrics
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        # Compute PSNR metric (no gradients needed for evaluation)
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_
        
        # Save detailed training statistics to CSV for later analysis
        with open(self.loss_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch,      # Current epoch number
                self.global_step,        # Total training steps so far
                loss.item(),             # Current loss value
                psnr_.item(),            # Current PSNR value
                log['lr'],               # Current learning rate
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp
            ])

        # Return loss for automatic backpropagation
        # PyTorch Lightning handles optimizer.zero_grad(), loss.backward(), optimizer.step()
        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},  # Show PSNR in progress bar
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        """
        Validate the model on a full image.
        
        Unlike training (which uses random ray batches), validation renders
        complete images to assess visual quality and compute image-level metrics.
        
        This allows us to:
        1. Visualize the rendered images vs ground truth
        2. Compute PSNR on full images (more meaningful than per-ray)
        3. Generate depth maps to understand scene geometry
        
        Args:
            batch: Full image worth of rays (H*W rays)
            batch_nb: Batch index (image number in validation set)
        
        Returns:
            Dictionary with validation loss and PSNR
        """
        # Extract rays and ground truth for full image
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 8) All rays for one image
        rgbs = rgbs.squeeze()  # (H*W, 3) All ground truth colors
        
        # Render the full image through the NeRF
        results = self(rays)
        
        # Compute validation loss
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        # Log images to TensorBoard for the first validation image
        # This lets us visually inspect: Ground Truth | Prediction | Depth Map
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            
            # Reshape flat ray outputs back to image format
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W) for TensorBoard
            
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # Ground truth
            
            # Visualize depth map (converts depth to RGB heatmap)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            
            # Stack all visualizations: [GT, Prediction, Depth]
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            
            # Log to TensorBoard
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        # Compute PSNR for this image
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        
        # Store for aggregation across all validation images
        self.validation_steps_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        """
        Aggregate validation metrics across all images.
        
        After validating on all images in the validation set,
        compute average metrics to track model performance over epochs.
        
        The checkpoint system uses these metrics to save the best models:
        - Models with lowest val_loss are saved automatically
        - This prevents overfitting by monitoring generalization
        """
        # 1. Collect all validation outputs from this epoch
        outputs = self.validation_steps_outputs
        
        # 2. Compute mean metrics across all validation images
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        # 3. Log aggregated metrics
        # These are used for:
        # - Progress bar display (shows validation quality during training)
        # - TensorBoard plots (track improvement over time)
        # - Model checkpointing (save best model based on val_loss)
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

        # 4. Clear storage for next validation epoch
        self.validation_steps_outputs.clear() 


if __name__ == '__main__':
    """
    Main training script entry point.
    
    COMPLETE TRAINING PIPELINE:
    ============================
    
    1. INITIALIZATION
       - Parse command-line arguments (dataset path, hyperparameters, etc.)
       - Create NeRF system (networks, loss, optimizers)
       - Set up logging and checkpointing
    
    2. DATA PREPARATION
       - Load images and camera poses
       - Precompute rays for all pixels in all images
       - Split into train/validation sets
    
    3. TRAINING LOOP (automated by PyTorch Lightning)
       For each epoch:
         For each batch of rays:
           a. FORWARD PASS:
              - Sample points along rays (coarse: 64, fine: 128 additional)
              - Apply positional encoding to 3D positions and directions
              - Query NeRF MLPs to get RGB + density (sigma)
              - Apply volume rendering equation:
                RGB(r) = Σ T_i * alpha_i * c_i
                Depth(r) = Σ T_i * alpha_i * z_i
           
           b. LOSS COMPUTATION:
              - MSE(predicted_rgb, ground_truth_rgb)
              - If fine network: loss_total = loss_coarse + loss_fine
           
           c. BACKPROPAGATION:
              - Compute gradients via automatic differentiation
              - Update network weights using Adam optimizer
              - Adjust learning rate via scheduler
       
         After epoch:
           - Run validation on held-out views
           - Save model checkpoint if val_loss improved
           - Log metrics to TensorBoard
    
    4. RENDERING (after training)
       - Novel view synthesis: Query trained NeRF from new camera poses
       - The network has learned a continuous 3D representation
       - Can render arbitrary views not seen during training
    
    KEY CONCEPTS:
    =============
    
    • Volume Rendering Equation:
      The core of NeRF. Integrates radiance and density along a ray:
      
      C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
      
      where:
      - r(t) = o + td is the ray (origin o, direction d)
      - σ is volume density (how opaque the point is)
      - c is emitted radiance (RGB color)
      - T(t) = exp(-∫ σ(s) ds) is accumulated transmittance
      
      Discrete approximation:
      C = Σ T_i · (1 - exp(-σ_i·δ_i)) · c_i
    
    • Hierarchical Sampling:
      - Coarse network: Uniform sampling → rough geometry
      - Fine network: Importance sampling → refine details
      - Sample more points where the ray passes through surfaces
    
    • Positional Encoding:
      Maps inputs to higher dimensions using Fourier features:
      γ(p) = [sin(2^0·πp), cos(2^0·πp), ..., sin(2^(L-1)·πp), cos(2^(L-1)·πp)]
      Helps MLPs learn high-frequency functions (fine details)
    """
    
    # ===== PARSE HYPERPARAMETERS =====
    hparams = get_opts()  # Load from command-line or use defaults
    
    # ===== CREATE NERF SYSTEM =====
    system = NeRFSystem(hparams)
    
    # ===== SETUP CHECKPOINTING =====
    # Save models during training to resume later or use the best version
    ckpt_dir = os.path.join('ckpts', hparams.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch:d}',      # Save as '0.ckpt', '1.ckpt', etc.
        monitor='val/loss',        # Track validation loss
        mode='min',                # Save models with minimum val_loss
        save_top_k=5               # Keep 5 best models
    )

    # ===== SETUP LOGGING =====
    # TensorBoard logs training curves, validation images, and metrics
    # View with: tensorboard --logdir=logs
    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    # ===== CONFIGURE TRAINING STRATEGY =====
    # Handle single-GPU, multi-GPU, or CPU training
    if hparams.num_gpus > 1:
        training_strategy = 'ddp'  # Distributed Data Parallel for multi-GPU
    elif hparams.num_gpus == 1:
        training_strategy = 'auto'  # Single GPU optimization
    else:
        training_strategy = 'auto'  # CPU fallback

    # ===== CREATE TRAINER =====
    # PyTorch Lightning Trainer automates the training loop:
    # - Handles device placement (CPU/GPU)
    # - Manages distributed training
    # - Gradient accumulation and clipping
    # - Automatic checkpointing
    # - Progress tracking and logging
    trainer = Trainer(
        max_epochs=hparams.num_epochs,  # Total training epochs (e.g., 16)
        callbacks=[checkpoint_callback], # Auto-save best models
        logger=logger,                   # Log to TensorBoard
        
        # --- Device Configuration ---
        accelerator='gpu' if hparams.num_gpus > 0 else 'cpu',
        devices=hparams.num_gpus if hparams.num_gpus > 0 else 1,
        strategy=training_strategy,      # Single/multi-GPU strategy
        
        enable_progress_bar=True,        # Show training progress
        
        # --- Training Options ---
        num_sanity_val_steps=1,          # Quick validation check before training
        benchmark=True,                  # Optimize CUDA kernels for speed
        profiler='simple' if hparams.num_gpus==1 else None  # Performance profiling
    )

    # ===== START TRAINING =====
    # This runs the complete training pipeline:
    # 1. Initialize data loaders
    # 2. Run training_step() for each batch
    # 3. Run validation_step() after each epoch
    # 4. Save checkpoints when val_loss improves
    # 5. Continue until max_epochs reached
    trainer.fit(system, ckpt_path=hparams.ckpt_path)  # Resume from checkpoint if provided