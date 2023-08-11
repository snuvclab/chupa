import os
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

class ImageLogger(Callback):
    def __init__(self, frequency, max_images, frequency_type="batch", 
                 clamp=True, rescale=True,
                 log_first_step=True, log_images_kwargs={}):
        super().__init__()
        self.freq = frequency
        self.max_images = max_images
        self.freq_type = frequency_type
        assert self.freq_type in ["batch", "epoch", "global_step"]

        self.logger_log_images = { TensorBoardLogger: self._testtube, }

        self.clamp = clamp
        self.rescale = rescale
        self.log_images_kwargs = log_images_kwargs
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = (make_grid(images[k]) + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            pl_module.logger.experiment.add_image(f"{split}/{k}", grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            filename = f"{k}_gs-{global_step+1:06}_e-{current_epoch+1:06}_b-{batch_idx+1:06}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if self.freq_type == "batch":
            check_idx = batch_idx
        elif self.freq_type == "epoch":
            check_idx = pl_module.current_epoch+1
        else:
            check_idx = pl_module.global_step

        is_train = pl_module.training
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images):
            # if self.check_frequency(check_idx) and self.max_images > 0:
            if check_idx % self.freq == 0 and self.max_images > 0:
                logger = type(pl_module.logger)

                if is_train:
                    pl_module.eval()

                with torch.no_grad():
                    images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

                for k in images:
                    images[k] = images[k][:min(len(images[k]), self.max_images)]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

                self.log_local(pl_module.logger.save_dir, split, images,
                            pl_module.global_step, pl_module.current_epoch, batch_idx)

                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, images, pl_module.global_step, split)

                if is_train:
                    pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq_type != "epoch":
            if pl_module.global_step > 0 or self.log_first_step:
                self.log_img(pl_module, batch, batch_idx, split="train")
        else:
            if pl_module.current_epoch > 0 or self.log_first_step:
                if trainer.num_training_batches == batch_idx+1:
                    self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.freq_type != "epoch":
            if pl_module.global_step > 0 or self.log_first_step:
                self.log_img(pl_module, batch, batch_idx, split="val")
        else:
            if pl_module.current_epoch > 0 or self.log_first_step:
                if trainer.num_val_batches == batch_idx+1:
                    self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

