import lightning as L

from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch
from datamodule.wlasl import WLASLDataModule
from models.i3d.i3d import InceptionI3d

if __name__ == "__main__":
    data_dir = "WLASL_2/WLASL2000"
    annotations_path = "WLASL/code/preprocess/nslt_2000.json"

    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="wlasl-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=400,
        # fast_dev_run=10,
    )
    model = InceptionI3d(num_classes=2000)
    data = WLASLDataModule(
        data_dir=data_dir,
        annotations=annotations_path,
    )
    # trainer.fit(model, datamodule=data, ckpt_path="last")
    trainer.test(model, datamodule=data)
