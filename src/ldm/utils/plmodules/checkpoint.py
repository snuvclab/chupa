import os
import pytorch_lightning as pl

class CheckpointEveryNEpochs(pl.Callback):
    """
    Save a checkpoint every N epochs, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self, save_epoch_frequency, prefix="", use_modelcheckpoint_filename=False):
        """
        Args:
            save_epoch_frequency: how often to save in epoch
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        n_batches = trainer.num_training_batches
        if ((epoch+1) % self.save_epoch_frequency == 0) and (n_batches == batch_idx + 1):
            if self.use_modelcheckpoint_filename:
                filename = f"epoch{epoch+1:06}.ckpt"
            else:
                filename = f"{self.prefix}epoch{epoch:06d}_gs{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)