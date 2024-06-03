import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .conv3d import Conv3dBlock
from .maxpool3d import MaxPool3dDynamicPadding
from .inception import InceptionBlock


class InceptionI3d(L.LightningModule):

    VALID_ENDPOINTS = [
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    ]

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        pretrained_backbone="weights/rgb_imagenet.pt",
    ):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._pretrained_backbone = pretrained_backbone
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {self._final_endpoint}")

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Conv3dBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=[7, 7, 7],
            stride=(2, 2, 2),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dDynamicPadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2)
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Conv3dBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[1, 1, 1],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Conv3dBlock(
            in_channels=64,
            out_channels=192,
            kernel_size=[3, 3, 3],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dDynamicPadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2)
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionBlock(
            in_channels=192,
            out_channels=[64, 96, 128, 16, 32, 32],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionBlock(
            in_channels=256,
            out_channels=[128, 128, 192, 32, 96, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dDynamicPadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionBlock(
            in_channels=480,
            out_channels=[192, 96, 208, 16, 48, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionBlock(
            in_channels=512,
            out_channels=[160, 112, 224, 24, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionBlock(
            in_channels=512,
            out_channels=[128, 128, 256, 24, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionBlock(
            in_channels=512,
            out_channels=[112, 144, 288, 32, 64, 64],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionBlock(
            in_channels=528,
            out_channels=[256, 160, 320, 32, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dDynamicPadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2)
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionBlock(
            in_channels=832,
            out_channels=[256, 160, 320, 32, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionBlock(
            in_channels=832,
            out_channels=[384, 192, 384, 48, 128, 128],
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"
        self.avg_pool = nn.AvgPool3d([2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Conv3dBlock(
            in_channels=1024,
            out_channels=self._num_classes,
            kernel_size=[1, 1, 1],
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()
        self.load_pretained_backbone()

    def build(self):
        for k, v in self.end_points.items():
            self.add_module(k, v)

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Conv3dBlock(
            in_channels=1024,
            out_channels=self._num_classes,
            kernel_size=[1, 1, 1],
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)

        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
            return logits

        return x

    def load_pretained_backbone(self):
        pretrained = torch.load(self._pretrained_backbone)
        # Remove the logits
        pretrained.pop("logits.conv3d.weight", None)
        pretrained.pop("logits.conv3d.bias", None)
        self.load_state_dict(pretrained, strict=False)

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        x = self.avg_pool(x)
        return x

    def accuracy(self, predictions, targets, top_k=1):
        vals, idxs = torch.topk(predictions, k=top_k, dim=1)
        targets = torch.argmax(targets, dim=1)
        matched = torch.eq(idxs, targets.view(-1, 1)).any(dim=1)
        return matched.float().mean()

    def training_step(self, batch, batch_idx):
        vid, labels, name = batch
        t = vid.size(2)

        logits = self(vid)
        logits = F.interpolate(logits, t, mode="linear")
        loc_loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.max(logits, dim=2)[0]  # Keep only the highest for each frame
        truth = torch.max(labels, dim=2)[0]
        pred_loss = F.binary_cross_entropy_with_logits(preds, truth)

        loss = 0.5 * loc_loss + 0.5 * pred_loss

        top_1_accuracy = self.accuracy(preds, truth, top_k=1)
        top_5_accuracy = self.accuracy(preds, truth, top_k=5)
        top_10_accuracy = self.accuracy(preds, truth, top_k=10)

        self.log_dict(
            {
                "train_loss": loss,
                "train_location_loss": loc_loss,
                "train_prediction_loss": pred_loss,
                "train_accuracy_top_1": top_1_accuracy,
                "train_accuracy_top_5": top_5_accuracy,
                "train_accuracy_top_10": top_10_accuracy,
            },
            sync_dist=True,
            batch_size=6,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        vid, labels, name = batch
        t = vid.size(2)

        logits = self(vid)
        logits = F.interpolate(logits, t, mode="linear")
        loc_loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.max(logits, dim=2)[0]
        truth = torch.max(labels, dim=2)[0]
        pred_loss = F.binary_cross_entropy_with_logits(preds, truth)

        loss = 0.5 * loc_loss + 0.5 * pred_loss

        top_1_accuracy = self.accuracy(preds, truth, top_k=1)
        top_5_accuracy = self.accuracy(preds, truth, top_k=5)
        top_10_accuracy = self.accuracy(preds, truth, top_k=10)

        self.log_dict(
            {
                "val_loss": loss,
                "val_location_loss": loc_loss,
                "val_prediction_loss": pred_loss,
                "val_accuracy_top_1": top_1_accuracy,
                "val_accuracy_top_5": top_5_accuracy,
                "val_accuracy_top_10": top_10_accuracy,
            },
            sync_dist=True,
            batch_size=6,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        vid, labels, name = batch
        t = vid.size(2)
        logits = self(vid)
        logits = F.interpolate(logits, t, mode="linear")
        loc_loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.max(logits, dim=2)[0]
        truth = torch.max(labels, dim=2)[0]
        pred_loss = F.binary_cross_entropy_with_logits(preds, truth)

        loss = 0.5 * loc_loss + 0.5 * pred_loss

        self.log("test_loss", loss, batch_size=6)
        return preds

    def predict_step(self, batch, batch_idx):
        vid, labels, name = batch
        t = vid.size(2)
        logits = self(vid)
        logits = F.interpolate(logits, t, mode="linear")
        preds = torch.max(logits, dim=2)[0]
        truth = torch.max(labels, dim=2)[0]

        pred_idx = preds.argmax()
        truth_idx = truth.argmax()


        return pred_idx, truth_idx, name

