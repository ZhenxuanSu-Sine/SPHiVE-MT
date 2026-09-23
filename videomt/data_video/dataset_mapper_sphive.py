import copy
import json
import os
import random

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["SPHiVEDatasetMapper", "load_sphive_manifest"]


def _resolve_path(root, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_sphive_manifest(path):
    """Load a SPHiVE JSONL manifest.

    Each line is one sample with a frames list. Relative image/supervision
    paths are resolved relative to the manifest directory.
    """
    root = os.path.dirname(os.path.abspath(path))
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "frames" not in sample or not sample["frames"]:
                raise ValueError(f"{path}:{line_no}: missing non-empty frames")
            for frame in sample["frames"]:
                frame["image"] = _resolve_path(root, frame["image"])
                frame["supervision"] = _resolve_path(
                    root, frame.get("supervision")
                )
            samples.append(sample)
    return samples


def _unpack_masks(data, key, height, width):
    if key not in data:
        return np.zeros((0, height, width), dtype=np.bool_)
    packed = np.asarray(data[key], dtype=np.uint8)
    if packed.ndim == 2:
        packed = packed[None]
    masks = np.unpackbits(packed, axis=-1)
    return masks[..., :width].astype(np.bool_, copy=False)


def _load_supervision(path):
    with np.load(path, allow_pickle=False) as data:
        height, width = [int(x) for x in data["mask_hw"]]
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        far = np.asarray(data["far"], dtype=np.bool_)
        track_ids = np.asarray(data["track_ids"], dtype=np.int64)
        masks = _unpack_masks(data, "masks_packed", height, width)
        n = len(node_ids)
        if not (len(far) == len(track_ids) == len(masks) == n):
            raise ValueError(
                f"{path}: node_ids/far/track_ids/masks must have the same length"
            )
        if "valid_mask_packed" in data:
            valid = _unpack_masks(data, "valid_mask_packed", height, width)
            valid_mask = valid[0] if len(valid) else np.ones((height, width), dtype=np.bool_)
        else:
            valid_mask = np.ones((height, width), dtype=np.bool_)
        exhaustive = np.asarray(
            data["exhaustive_node_ids"] if "exhaustive_node_ids" in data else np.empty(0, dtype=np.int64),
            dtype=np.int64,
        )
    return {
        "node_ids": node_ids,
        "far": far,
        "track_ids": track_ids,
        "masks": masks,
        "valid_mask": valid_mask,
        "exhaustive_node_ids": exhaustive,
    }


class SPHiVEDatasetMapper:
    """Mapper for the unified SPHiVE image/video format."""

    @configurable
    def __init__(self, is_train, *, augmentations, image_format, sampling_frame_num):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.sampling_frame_num = sampling_frame_num

    @classmethod
    def from_config(cls, cfg, is_train=True, **kwargs):
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                T.RandomFlip(),
            ]
        else:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    "choice",
                )
            ]
        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
        }

    def _select_indices(self, frames):
        length = len(frames)
        if not self.is_train:
            return list(range(length))
        n = self.sampling_frame_num
        if n <= 0:
            return list(range(length))
        if length == n:
            return list(range(length))
        if length < n:
            # Current VidEoMT training expects fixed T. Use n=1 for true single-frame training.
            return list(range(length)) + [length - 1] * (n - length)
        supervised = [
            i for i, frame in enumerate(frames)
            if frame.get("supervision") is not None
        ]
        if supervised:
            anchor = random.choice(supervised)
            min_start = max(0, anchor - n + 1)
            max_start = min(anchor, length - n)
            start = random.randint(min_start, max_start)
        else:
            start = random.randint(0, length - n)
        return list(range(start, start + n))

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        frames = dataset_dict["frames"]
        indices = self._select_indices(frames)
        images = []
        targets = []
        transforms = None
        raw_height = raw_width = None
        for out_i, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            image = utils.read_image(frame["image"], format=self.image_format)
            if out_i == 0:
                raw_height, raw_width = image.shape[:2]
                aug_input = T.AugInput(image)
                transforms = self.augmentations(aug_input)
                image = aug_input.image
            else:
                image = transforms.apply_image(image)
            images.append(
                torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            )
            supervision_path = frame.get("supervision")
            if supervision_path is None:
                targets.append(None)
                continue
            target = _load_supervision(supervision_path)
            transformed_masks = []
            for mask in target["masks"]:
                transformed = transforms.apply_segmentation(mask.astype(np.uint8))
                transformed_masks.append(torch.from_numpy(transformed > 0))
            if transformed_masks:
                masks_tensor = torch.stack(transformed_masks, dim=0)
            else:
                h, w = image.shape[:2]
                masks_tensor = torch.zeros((0, h, w), dtype=torch.bool)
            valid_mask = transforms.apply_segmentation(target["valid_mask"].astype(np.uint8))
            valid_mask = torch.from_numpy(valid_mask > 0)
            targets.append({
                "node_ids": torch.as_tensor(target["node_ids"], dtype=torch.long),
                "far": torch.as_tensor(target["far"], dtype=torch.bool),
                "track_ids": torch.as_tensor(target["track_ids"], dtype=torch.long),
                "masks": masks_tensor,
                "valid_mask": valid_mask,
                "exhaustive_node_ids": torch.as_tensor(
                    target["exhaustive_node_ids"], dtype=torch.long
                ),
            })
        return {
            "image": images,
            "sphive_targets": targets,
            "frame_idx": indices,
            "height": raw_height,
            "width": raw_width,
        }