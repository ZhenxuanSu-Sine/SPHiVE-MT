# SPHiVE-MT supervision contract

SPHiVE-MT separates temporal context from supervision. A frame may be used by
the video model even when no segmentation loss is allowed on that frame.

## Frame-level supervision

`gt_valid: List[bool]`, length `T`.

- `True`: the frame contains usable annotation.
- `False`: the frame is context-only.
- Context-only frames still participate in the forward pass and query
  propagation, but they do not contribute matching, classification, BCE, or
  Dice loss.

This is the primary mechanism for dense raw video with sparse GT, e.g. 20 Hz
images with 2/5/10/20 Hz annotation.

## Pixel-level supervision

`pixel_valid_masks: BoolTensor[T, H, W]`.

- `True`: the pixel is supervised.
- `False`: the pixel is unknown / ignored, not background.

The validity mask is applied to both Hungarian mask costs and BCE/Dice losses.
For semantic datasets, it is derived from the dataset ignore label. Frames with
no valid pixels are treated as context-only.

## Exhaustive-label supervision

`label_exhaustive: List[bool]`, length `T`.

This controls whether unmatched queries may be trained as `no-object`.

- `True`: the annotation is exhaustive enough to treat unmatched queries as
  negatives.
- `False`: unmatched queries are left unsupervised; only matched queries
  receive classification supervision.

Semantic frames containing ignore/unknown pixels default to non-exhaustive.
Fully annotated VIS frames remain exhaustive by default.

This distinction is required for partial annotations: an unannotated object or
region must not become an implicit negative example.

## Relationship

The three masks are intentionally orthogonal:

1. `gt_valid` answers: should this frame contribute any loss?
2. `pixel_valid_masks` answers: which pixels may contribute mask loss?
3. `label_exhaustive` answers: may unmatched queries be called background?

Future hierarchical / partial-label supervision should build on this contract
rather than encoding unknown labels as background.
