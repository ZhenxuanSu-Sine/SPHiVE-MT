import torch
import torch.nn.functional as F

from detectron2.utils.comm import get_world_size

from .criterion_videomt import VideoSetCriterion
from .utils.misc import is_dist_avail_and_initialized


class SPHiVESetCriterion(VideoSetCriterion):
    """SPHiVE loss: instance set loss + taxonomy semantic projection loss."""

    def __init__(self, *args, descendant_matrix, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "descendant_matrix", descendant_matrix.to(torch.bool), persistent=False
        )

    def _filter_indices_by_frame_valid(self, targets, indices):
        """Keep only supervised and visible instance targets."""
        filtered = []
        for target, (src_idx, tgt_idx) in zip(targets, indices):
            if not self._frame_is_valid(target) or tgt_idx.numel() == 0:
                filtered.append((src_idx[:0], tgt_idx[:0]))
                continue
            ids = target["ids"][tgt_idx].reshape(-1)
            keep = ids != -1
            filtered.append((src_idx[keep], tgt_idx[keep]))
        return filtered

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Matched-only hierarchical classification loss.

        A target at node u accepts predictions at u or any descendant of u.
        Unmatched queries are intentionally not forced to no-object because
        far/partial regions may contain valid but non-instantiated entities.
        """
        logits = outputs["pred_logits"].float()
        probs = logits.softmax(-1)[..., : self.num_classes]
        losses = []
        for b, (target, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if src_idx.numel() == 0:
                continue
            target_nodes = target["labels"][tgt_idx].long()
            allowed = self.descendant_matrix[target_nodes].to(probs)
            subtree_prob = (probs[b, src_idx] * allowed).sum(-1)
            losses.append(-torch.log(subtree_prob.clamp_min(1e-8)))
        if not losses:
            return {"loss_ce": logits.sum() * 0.0}
        return {"loss_ce": torch.cat(losses).mean()}

    def loss_semantic_projection(self, outputs, targets):
        """Project all queries to exhaustive taxonomy nodes and supervise unions."""
        logits = outputs["pred_logits"].float()
        class_probs = logits.softmax(-1)[..., : self.num_classes]
        mask_probs = outputs["pred_masks"].float().sigmoid()[:, :, 0]

        bce_terms = []
        dice_terms = []
        for b, target in enumerate(targets):
            if not self._frame_is_valid(target):
                continue
            exhaustive = target.get("exhaustive_node_ids")
            if exhaustive is None or exhaustive.numel() == 0:
                continue

            region_nodes = target.get("region_node_ids")
            region_masks = target.get("region_masks")
            pixel_valid = target.get("pixel_valid")

            pred_h, pred_w = mask_probs.shape[-2:]
            if pixel_valid is None:
                valid = torch.ones(
                    (pred_h, pred_w), dtype=torch.bool, device=mask_probs.device
                )
            else:
                valid = F.interpolate(
                    pixel_valid[None].float().to(mask_probs),
                    size=(pred_h, pred_w),
                    mode="nearest",
                )[0, 0].bool()

            if region_masks is not None and region_masks.numel() > 0:
                gt_regions = F.interpolate(
                    region_masks.float().to(mask_probs),
                    size=(pred_h, pred_w),
                    mode="nearest",
                )[:, 0].bool()
            else:
                gt_regions = None

            for node in exhaustive.long():
                subtree = self.descendant_matrix[node].to(class_probs.device)
                query_prob = class_probs[b, :, subtree].sum(-1)
                contribution = query_prob[:, None, None] * mask_probs[b]
                pred = 1.0 - torch.prod(
                    1.0 - contribution.clamp(0.0, 1.0 - 1e-6), dim=0
                )

                gt = torch.zeros_like(pred, dtype=torch.bool)
                if gt_regions is not None:
                    include = self.descendant_matrix[
                        node, region_nodes.long().to(self.descendant_matrix.device)
                    ]
                    if include.any():
                        gt = gt_regions[include.to(gt_regions.device)].any(dim=0)

                if not valid.any():
                    continue
                pred_v = pred[valid].clamp(1e-6, 1.0 - 1e-6)
                gt_v = gt[valid].to(pred_v)
                bce_terms.append(F.binary_cross_entropy(pred_v, gt_v))

                numerator = 2.0 * (pred_v * gt_v).sum()
                denominator = pred_v.sum() + gt_v.sum()
                dice_terms.append(1.0 - (numerator + 1.0) / (denominator + 1.0))

        zero = outputs["pred_masks"].sum() * 0.0
        loss_bce = torch.stack(bce_terms).mean() if bce_terms else zero
        loss_dice = torch.stack(dice_terms).mean() if dice_terms else zero
        return {
            "loss_semantic_mask": loss_bce,
            "loss_semantic_dice": loss_dice,
        }

    def forward(self, outputs, targets, matcher_outputs=None, ret_match_result=False):
        if matcher_outputs is None:
            match_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        else:
            match_outputs = {k: v for k, v in matcher_outputs.items() if k != "aux_outputs"}

        indices = self.matcher(match_outputs, targets)
        loss_indices = self._filter_indices_by_frame_valid(targets, indices)

        num_masks = sum(
            int((target["ids"].reshape(-1) != -1).sum().item())
            for target in targets
            if self._frame_is_valid(target)
        )
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, loss_indices, num_masks)
            )
        losses.update(self.loss_semantic_projection(outputs, targets))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_indices = self._filter_indices_by_frame_valid(targets, aux_indices)
                for loss in self.losses:
                    loss_dict = self.get_loss(
                        loss, aux_outputs, targets, aux_indices, num_masks
                    )
                    losses.update({k + f"_{i}": v for k, v in loss_dict.items()})

        if ret_match_result:
            return losses, indices
        return losses