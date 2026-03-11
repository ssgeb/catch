"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

try:
    from detectron2.utils.memory import retry_if_cuda_oom
except ImportError:
    def retry_if_cuda_oom(func):
        return func

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        mask_threshold=0.5,
        score_with_mask=True,
        mask_process_chunk=32,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.mask_threshold = mask_threshold
        self.score_with_mask = score_with_mask
        self.mask_process_chunk = mask_process_chunk
        self.deploy_mode = False
        self._mask_postprocess = retry_if_cuda_oom(self._mask_postprocess_single_image)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def _mask_postprocess_single_image(self, mask_logits: torch.Tensor, height: int, width: int):
        mask_chunks = []
        mask_score_chunks = [] if self.score_with_mask else None

        for mask_chunk in mask_logits.split(self.mask_process_chunk, dim=0):
            mask_chunk = F.interpolate(
                mask_chunk[:, None],
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )[:, 0]
            mask_probs = mask_chunk.sigmoid()

            if self.score_with_mask:
                # Match MaskDINO instance scoring: average sigmoid probability on positive pixels.
                binary_masks = (mask_probs > self.mask_threshold).to(mask_chunk.dtype)
                mask_scores = (
                    (mask_probs.flatten(1) * binary_masks.flatten(1)).sum(1)
                    / (binary_masks.flatten(1).sum(1) + 1e-6)
                )
                mask_score_chunks.append(mask_scores.cpu())

            mask_chunks.append(mask_probs.cpu())

        masks = torch.cat(mask_chunks, dim=0)
        mask_scores = torch.cat(mask_score_chunks, dim=0) if self.score_with_mask else None
        return masks, mask_scores

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        mask_logits = outputs.get('pred_masks')
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        boxes = bbox_pred
        gathered_masks = None

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            if mask_logits is not None:
                gathered_masks = mask_logits.gather(
                    dim=1,
                    index=index[:, :, None, None].repeat(1, 1, mask_logits.shape[-2], mask_logits.shape[-1]),
                )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                if mask_logits is not None:
                    gathered_masks = mask_logits.gather(
                        dim=1,
                        index=index[:, :, None, None].repeat(1, 1, mask_logits.shape[-2], mask_logits.shape[-1]),
                    )
            elif mask_logits is not None:
                gathered_masks = mask_logits

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for batch_idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab.cpu(), boxes=box.cpu(), scores=sco.cpu())
            if gathered_masks is not None:
                width = int(orig_target_sizes[batch_idx, 0].item())
                height = int(orig_target_sizes[batch_idx, 1].item())
                masks, mask_scores = self._mask_postprocess(gathered_masks[batch_idx], height, width)
                if self.score_with_mask:
                    result["scores"] = result["scores"] * mask_scores.to(result["scores"].dtype)
                result["masks"] = masks
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
