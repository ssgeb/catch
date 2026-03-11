"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from .mask_utils import batch_dice_loss, batch_sigmoid_ce_loss, point_sample

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000,
                num_sample_points=4096):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_mask = weight_dict.get('cost_mask', 0.0)
        self.cost_dice = weight_dict.get('cost_dice', 0.0)

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.num_sample_points = num_sample_points

        assert (
            self.cost_class != 0
            or self.cost_bbox != 0
            or self.cost_giou != 0
            or self.cost_mask != 0
            or self.cost_dice != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        has_mask_targets = all("masks" in target for target in targets)
        has_mask_preds = "pred_masks" in outputs
        use_mask_cost = has_mask_targets and has_mask_preds and (self.cost_mask != 0 or self.cost_dice != 0)

        cost_matrices = []
        indices_pre = []

        for batch_idx in range(bs):
            if self.use_focal_loss:
                out_prob = F.sigmoid(outputs["pred_logits"][batch_idx])
            else:
                out_prob = outputs["pred_logits"][batch_idx].softmax(-1)

            out_bbox = outputs["pred_boxes"][batch_idx]
            tgt_ids = targets[batch_idx]["labels"]
            tgt_bbox = targets[batch_idx]["boxes"]
            num_classes = out_prob.shape[-1]

            if tgt_bbox.numel() == 0:
                empty = out_bbox.new_zeros((num_queries, 0)).cpu()
                cost_matrices.append(empty)
                indices_pre.append((np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)))
                continue

            if tgt_ids.numel() > 0:
                invalid = (tgt_ids < 0) | (tgt_ids >= num_classes)
                if invalid.any():
                    bad_labels = torch.unique(tgt_ids[invalid]).detach().cpu().tolist()
                    image_id = targets[batch_idx].get("image_id", None)
                    if torch.is_tensor(image_id):
                        image_id = image_id.detach().cpu().flatten().tolist()
                    raise ValueError(
                        f"Target label out of range for matcher: labels={bad_labels}, "
                        f"valid=[0, {num_classes - 1}], image_id={image_id}. "
                        "Check num_classes/remap_mscoco_category/category_id mapping."
                    )

            if self.change_matcher and epoch >= self.matcher_change_epoch:
                class_score = out_prob[:, tgt_ids]
                bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
                cost_class = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
                cost_bbox = out_bbox.new_zeros(cost_class.shape)
                cost_giou = out_bbox.new_zeros(cost_class.shape)
            else:
                if self.use_focal_loss:
                    cls_prob = out_prob[:, tgt_ids]
                    neg_cost_class = (1 - self.alpha) * (cls_prob ** self.gamma) * (-(1 - cls_prob + 1e-8).log())
                    pos_cost_class = self.alpha * ((1 - cls_prob) ** self.gamma) * (-(cls_prob + 1e-8).log())
                    cost_class = pos_cost_class - neg_cost_class
                else:
                    cost_class = -out_prob[:, tgt_ids]

                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )

            if use_mask_cost:
                out_mask = outputs["pred_masks"][batch_idx][:, None]
                tgt_mask = targets[batch_idx]["masks"].to(device=out_mask.device, dtype=out_mask.dtype)[:, None]
                point_coords = torch.rand(1, self.num_sample_points, 2, device=out_mask.device)
                sampled_tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
                sampled_out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
                cost_mask = batch_sigmoid_ce_loss(sampled_out_mask.float(), sampled_tgt_mask.float())
                cost_dice = batch_dice_loss(sampled_out_mask.float(), sampled_tgt_mask.float())
            else:
                cost_mask = out_bbox.new_zeros(cost_class.shape)
                cost_dice = out_bbox.new_zeros(cost_class.shape)

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            C = torch.nan_to_num(C, nan=1.0).cpu()
            cost_matrices.append(C)
            indices_pre.append(linear_sum_assignment(C))

        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(cost_matrices, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, cost_matrices, k=1, initial_indices=None):
        indices_list = []
        work_matrices = [c.clone() for c in cost_matrices]
        for i in range(k):
            indices_k = [linear_sum_assignment(c) for c in work_matrices] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(work_matrices, indices_k):
                if len(idx_k[0]) == 0:
                    continue
                idx_k = np.stack(idx_k)
                c[idx_k[0], idx_k[1]] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(cost_matrices))]
        return indices_list
