from pytorch_lightning.callbacks import Callback

class StopAtEpoch(Callback):
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.stop_epoch:
            print(f"Stopping training at epoch {trainer.current_epoch}")
            trainer.should_stop = True
