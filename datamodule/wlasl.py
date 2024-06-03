import json
import math
import os
import random
import multiprocessing
import cv2
import lightning as L

import numpy as np
from sympy import nfloat
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils.videotransforms as vt


class WLASLDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, annotations: str, batch_size=6):
        super().__init__()
        self.data_dir = data_dir
        self.annotations = annotations
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose(
            [
                vt.RandomCrop(224),
                vt.RandomHorizontalFlip(),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                vt.CenterCrop(224),
            ]
        )

    def setup(self, stage):
        if stage in ("fit", None):
            self.train_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="train",
                transforms=self.train_transforms,
            )
            self.val_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="val",
                transforms=self.val_transforms,
            )
        elif stage == "test":
            self.test_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="test",
                transforms=self.val_transforms,
            )
        elif stage == "predict":
            self.predict_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="train",
                transforms=self.train_transforms,
                max_size=1000,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )


class WLASLDataset(Dataset):
    def __init__(
        self, annotations_path: str, data_dir: str, subset: str, max_size=None, transforms=None
    ):
        self.data_dir = data_dir
        self.annotations_path = annotations_path
        self.transforms = transforms
        self.n_classes = 2000
        self.frames_per_clip = 64
        self.max_size = max_size

        assert subset in [
            "train",
            "val",
            "test",
        ], f"subset must be one of ['train', 'val', 'test']"

        self.subset = subset
        self.data = self._prepare_dataset(annotations_path, data_dir, subset, max_size)

    def _prepare_dataset(self, annotations_path: str, data_dir: str, subset: str, max_size):
        dataset = []

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        count = 0

        for k, v in annotations.items():
            if max_size and count >= max_size:
                break
            if v["subset"] != subset:
                continue

            video_path = os.path.join(data_dir, f"{k}.mp4")

            if not os.path.exists(video_path):
                continue
            n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

            action = v["action"][0]
            frame_start = v["action"][1]
            frame_end = v["action"][2]
            frame_count = frame_end - frame_start

            label = np.zeros((self.n_classes, n_frames), dtype=np.float32)
            for l in range(n_frames):
                label[action][l] = 1

            dataset.append(
                {
                    "name": k,
                    "video_path": video_path,
                    "action": label,
                    "start": (
                        frame_start
                        if len(k) == 6 and self.subset in ["train", "val"]
                        else 0
                    ),
                    "frames": (
                        frame_count if self.subset in ["train", "val"] else n_frames
                    ),
                }
            )
            count += 1

        return dataset

    def __len__(self):
        return len(self.data)

    def _video_to_tensor(self, video_path, start_f=0, count=64):
        vid = cv2.VideoCapture(video_path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []

        while True:
            success, frame = vid.read()
            if not success:
                break

            if len(frames) >= count:
                break

            w, h, c = frame.shape
            if w < 226 or h < 226:
                d = 226.0 - min(w, h)
                sc = 1 + d / min(w, h)
                frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                new_h = math.ceil(h * (256 / h))
                new_w = math.ceil(w * (256 / w))
                frame = cv2.resize(frame, dsize=(new_w, new_h))

            frame = (frame / 255.0) * 2 - 1

            frames.append(frame)

        np_frames = np.asarray(frames, dtype=np.float32)

        # Pad to same size
        if self.subset in ["train", "val"] and count > 0:
            np_frames = self._pad_video(np_frames, count)

        if self.transforms:
            np_frames = self.transforms(np_frames)

        return torch.from_numpy(np_frames.transpose(3, 0, 1, 2))  # C, T, H, W

    def _pad_video(self, video, count):
        n_frames, h, w, c = video.shape
        if n_frames >= count:
            return video

        n_pad = count - n_frames

        if not n_pad:
            return video

        prob = np.random.random_sample()
        if prob > 0.5:
            pad = np.tile(np.expand_dims(video[0], axis=0), (n_pad, 1, 1, 1))
        else:
            pad = np.tile(np.expand_dims(video[-1], axis=0), (n_pad, 1, 1, 1))

        padded_video = np.concatenate([pad, video], axis=0)

        return padded_video

    def _tensor_label(self, label, count):
        # if self.subset in ["test"]:
        #     return torch.from_numpy(label)
        label = label[:, 0]
        label = np.tile(label, (count, 1)).transpose((1, 0))
        return torch.from_numpy(label)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item["name"]
        video_path = item["video_path"]
        action = item["action"]
        frames_count = item["frames"]
        start = item["start"]

        try:
            start_f = random.randint(0, frames_count - self.frames_per_clip - 1) + start
        except ValueError:
            start_f = start

        vid = self._video_to_tensor(video_path, start_f, self.frames_per_clip)
        label_tensor = self._tensor_label(action, self.frames_per_clip)

        return vid, label_tensor, name
