import argparse
import os

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from train_net_video import Trainer
from videomt import (
    SPHiVEDatasetMapper,
    add_videomt_config,
    load_sphive_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal SPHiVE inference.")
    parser.add_argument("--config-file", default="configs/sphive/example.yaml")
    parser.add_argument("--weights", default="model.pth")
    parser.add_argument("--manifest", default="datasets/sphive/val.jsonl")
    parser.add_argument("--output", default="output/sphive_inference")
    parser.add_argument("--save-semantic-scores", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.BACKBONE.TEST.TASK = "sphive"
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    mapper = SPHiVEDatasetMapper(cfg, is_train=False)
    samples = load_sphive_manifest(args.manifest)
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for index, sample in enumerate(samples):
            mapped = mapper(sample)
            result = model([mapped])
            if not args.save_semantic_scores:
                result.pop("semantic_scores", None)
            torch.save(result, os.path.join(args.output, f"{index:06d}.pt"))


if __name__ == "__main__":
    main()
