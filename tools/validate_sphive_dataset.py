import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SPHiVE manifest/taxonomy files.")
    parser.add_argument("--manifest", default="datasets/sphive/train.jsonl")
    parser.add_argument("--taxonomy", default="datasets/sphive/taxonomy.json")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    nodes = taxonomy["nodes"]
    node_ids = sorted(int(node["id"]) for node in nodes)
    if node_ids != list(range(len(node_ids))):
        raise ValueError("taxonomy node ids must be contiguous: 0..N-1")
    num_nodes = len(node_ids)

    root = os.path.dirname(os.path.abspath(args.manifest))
    samples = 0
    supervised_frames = 0

    with open(args.manifest, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            frames = sample.get("frames", [])
            if not frames:
                raise ValueError(f"line {line_no}: frames is empty")
            samples += 1

            for frame in frames:
                image = frame["image"]
                image = image if os.path.isabs(image) else os.path.join(root, image)
                if not os.path.exists(image):
                    raise FileNotFoundError(image)

                supervision = frame.get("supervision")
                if supervision is None:
                    continue
                supervision = supervision if os.path.isabs(supervision) else os.path.join(root, supervision)
                if not os.path.exists(supervision):
                    raise FileNotFoundError(supervision)

                supervised_frames += 1
                with np.load(supervision, allow_pickle=False) as data:
                    h, w = [int(x) for x in data["mask_hw"]]
                    ids = np.asarray(data["node_ids"], dtype=np.int64)
                    far = np.asarray(data["far"], dtype=np.bool_)
                    tracks = np.asarray(data["track_ids"], dtype=np.int64)
                    packed = np.asarray(data["masks_packed"], dtype=np.uint8)

                    if not (len(ids) == len(far) == len(tracks) == len(packed)):
                        raise ValueError(f"{supervision}: annotation arrays have different lengths")
                    if len(ids) and (ids.min() < 0 or ids.max() >= num_nodes):
                        raise ValueError(f"{supervision}: node id outside taxonomy")
                    if packed.ndim != 3 or packed.shape[1] != h or packed.shape[2] != (w + 7) // 8:
                        raise ValueError(f"{supervision}: invalid masks_packed shape")
                    if np.any(far & (tracks >= 0)):
                        raise ValueError(f"{supervision}: far annotations must use track_id=-1")

                    exhaustive = np.asarray(
                        data["exhaustive_node_ids"] if "exhaustive_node_ids" in data else [],
                        dtype=np.int64,
                    )
                    if len(exhaustive) and (
                        exhaustive.min() < 0 or exhaustive.max() >= num_nodes
                    ):
                        raise ValueError(f"{supervision}: exhaustive node outside taxonomy")

    print(f"OK: {samples} samples, {supervised_frames} supervised frames, {num_nodes} taxonomy nodes")


if __name__ == "__main__":
    main()
