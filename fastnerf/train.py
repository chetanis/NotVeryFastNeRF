import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.validation_steps_outputs = []

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.validation_steps_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        # 1. Stack the metrics from all batches in the epoch
        outputs = self.validation_steps_outputs
        
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        # 2. Use self.log() to record the metrics
        # The new API automatically handles progress bar and standard logging via this method.
        # 'prog_bar=True' adds it to the progress bar display.
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

        # 3. Clear the storage list for the next epoch
        self.validation_steps_outputs.clear() 


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    ckpt_dir = os.path.join('ckpts', hparams.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch:d}',
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    if hparams.num_gpus > 1:
        training_strategy = 'ddp'
    elif hparams.num_gpus == 1:
        training_strategy = 'auto' # Let lightning choose the best single-device strategy
    else:
        training_strategy = 'auto' # CPU or other fallback

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        
        # --- Updated Arguments for modern Lightning ---
        accelerator='gpu' if hparams.num_gpus > 0 else 'cpu', # Use GPU accelerator if available
        devices=hparams.num_gpus if hparams.num_gpus > 0 else 1, # Set number of devices
        strategy=training_strategy,                              # Set distributed strategy (ddp, etc.)
        
        enable_progress_bar=True, # Changed from integer 1 to boolean True
        
        # --- Remaining Arguments ---
        num_sanity_val_steps=1,
        benchmark=True,
        profiler='simple' if hparams.num_gpus==1 else None # Modern profiler argument takes a string
    )

    trainer.fit(system,ckpt_path=hparams.ckpt_path)