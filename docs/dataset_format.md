# Dataset format

SPHiVE-MT uses one format for images and videos. A sample is always a sequence of frames.

```json
{
  "frames": [
    {"image": "images/000000.jpg", "supervision": null},
    {"image": "images/000001.jpg", "supervision": "labels/000001.npz"},
    {"image": "images/000002.jpg", "supervision": null}
  ]
}
```

- A single image is a sequence with `T=1`.
- A single-frame label may also be trained with surrounding unlabeled frames.
- `supervision: null` means context-only: the frame participates in the forward pass but contributes no loss.

## Frame supervision

Each supervised frame uses one `.npz` file:

```text
mask_hw                  int32 [2]
valid_mask_packed        uint8 [H, ceil(W/8)]   # optional

node_ids                 int32 [N]
far                      bool  [N]
track_ids                int64 [N]
masks_packed             uint8 [N, H, ceil(W/8)]

exhaustive_node_ids      int32 [K]
```

Masks are binary and packed with `numpy.packbits`.

### Annotation

Each annotation is:

```text
(node_id, mask, far, track_id)
```

- `node_id`: taxonomy node ID.
- `mask`: region mask.
- `far=false`: instance-level annotation.
- `far=true`: semantic-only region; instance decomposition is unknown.
- `track_id>=0`: cross-frame identity.
- `track_id=-1`: no temporal identity.

`far` does not change semantic meaning. All annotations contribute to semantic supervision.

Example:

```text
regular_vehicle, car_A,      far=false, track_id=17
regular_vehicle, car_B,      far=false, track_id=23
vehicle,         distant_blob, far=true, track_id=-1
```

The semantic `vehicle` region is the union of all three masks after taxonomy projection.

A pure semantic dataset is simply a dataset where all annotations are `far=true`.

## Partial labels and hierarchy

Partial labels are represented by the taxonomy depth of `node_id`.

If an old dataset only knows `vehicle`, it annotates `vehicle`; it does not guess a child such as `regular_vehicle`.

Parent and child masks may overlap.

## Exhaustive nodes

`exhaustive_node_ids` tells which taxonomy nodes are completely annotated inside the valid area.

If `vehicle` is exhaustive but no vehicle annotation exists, `vehicle` is a known negative.

Exhaustiveness is node-specific and does not automatically propagate to parents or children.

## Ignore / unknown

`valid_mask_packed` marks pixels where supervision is valid.

- valid = 1: supervision may be used.
- valid = 0: unknown / ignore, not background.

If omitted, the whole frame is valid.


## Taxonomy

Taxonomy is a separate JSON file. Node ids must be contiguous from 0 to N-1.

```json
{
  "nodes": [
    {"id": 0, "name": "traffic_participant", "parent": null},
    {"id": 1, "name": "vehicle", "parent": 0},
    {"id": 2, "name": "regular_vehicle", "parent": 1}
  ]
}
```

The model uses the tree to expand parent supervision to its subtree. A label on `vehicle` does not guess which child is correct.

## Training

Use `DATASET_TYPE: ["sphive"]`, set `DATASETS.TRAIN` to the JSONL manifest path, and set `DATASETS.TAXONOMY_FILE` to the taxonomy JSON.

Current VidEoMT training uses a fixed temporal length per run:

- true single-frame training: `INPUT.SAMPLING_FRAME_NUM: 1`
- sparse video supervision: use `T>1`; unlabeled frames keep `supervision: null`

## Minimal inference output

`TEST.TASK: "sphive"` returns:

- query scores / labels / masks / ids
- overlapping semantic score maps for every taxonomy node

Panoptic conflict resolution is intentionally left for later.
