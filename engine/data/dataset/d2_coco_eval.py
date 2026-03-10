"""
Detectron2-based COCO evaluator wrapper with DEIMv2 evaluator interface.
"""

import copy
import inspect
from typing import Dict, List

import numpy as np
import torch
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval

from ...core import register

__all__ = ["Detectron2CocoEvaluator"]


@register()
class Detectron2CocoEvaluator(object):
    """
    A lightweight wrapper that reuses detectron2 COCO evaluation logic while
    keeping the same methods used by DEIMv2 training/eval pipeline.
    """

    def __init__(self, coco_gt, iou_types, use_fast_impl=True, max_dets_per_image=None):
        assert isinstance(iou_types, (list, tuple)), "iou_types should be a list/tuple."
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = list(iou_types)
        self.use_fast_impl = use_fast_impl
        self.max_dets_per_image = max_dets_per_image

        self.cleanup()

    def cleanup(self):
        self.img_ids = []
        self._predictions = {k: [] for k in self.iou_types}
        self.coco_eval = {}

    def update(self, predictions: Dict[int, Dict[str, torch.Tensor]]):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            self._predictions[iou_type].extend(self.prepare(predictions, iou_type))

    def synchronize_between_processes(self):
        self.img_ids = _merge_img_ids(self.img_ids)
        for iou_type in self.iou_types:
            self._predictions[iou_type] = _merge_predictions(self._predictions[iou_type])

    def accumulate(self):
        try:
            from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
        except ImportError as exc:
            raise ImportError(
                "Detectron2CocoEvaluator requires detectron2. "
                "Please install detectron2 before using this evaluator."
            ) from exc

        for iou_type in self.iou_types:
            results = self._predictions[iou_type]
            eval_kwargs = {}
            supported_params = inspect.signature(_evaluate_predictions_on_coco).parameters
            if "use_fast_impl" in supported_params:
                eval_kwargs["use_fast_impl"] = self.use_fast_impl
            elif "cocoeval_fn" in supported_params:
                # DEIMv2 initializes faster_coco_eval as pycocotools, which breaks
                # detectron2's COCOeval_opt fast path. Force the compatible eval API.
                eval_kwargs["cocoeval_fn"] = COCOeval
            if "img_ids" in supported_params:
                eval_kwargs["img_ids"] = self.img_ids
            if "max_dets_per_image" in supported_params:
                eval_kwargs["max_dets_per_image"] = _normalize_max_dets_per_image(
                    self.max_dets_per_image
                )
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self.coco_gt,
                    results,
                    iou_type,
                    **eval_kwargs,
                )
                if len(results) > 0
                else None
            )
            self.coco_eval[iou_type] = coco_eval

    def summarize(self):
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval.get(iou_type)
            print(f"IoU metric: {iou_type}")
            if coco_eval is None:
                print("No predictions to evaluate.")
                continue
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            if "masks" not in prediction:
                raise KeyError(
                    "segm evaluation requested, but model predictions do not contain `masks`."
                )

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            masks = prediction["masks"]

            if isinstance(masks, torch.Tensor):
                if masks.ndim == 4:
                    masks = masks[:, 0]
                masks = (masks > 0.5).to(torch.uint8).cpu().numpy()
            else:
                masks = np.asarray(masks)
                if masks.ndim == 4:
                    masks = masks[:, 0]
                masks = (masks > 0.5).astype(np.uint8)

            rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype=np.uint8))[0] for mask in masks]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes: torch.Tensor):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _merge_img_ids(img_ids: List[int]):
    from ...misc import dist_utils
    all_img_ids = dist_utils.all_gather(img_ids)
    merged = []
    for item in all_img_ids:
        merged.extend(item)
    merged = np.array(merged)
    merged, _ = np.unique(merged, return_index=True)
    return merged.tolist()


def _merge_predictions(predictions: List[dict]):
    from ...misc import dist_utils
    all_predictions = dist_utils.all_gather(predictions)
    merged = []
    for item in all_predictions:
        merged.extend(item)
    return merged


def _normalize_max_dets_per_image(max_dets_per_image):
    if isinstance(max_dets_per_image, int):
        return [1, 10, max_dets_per_image]
    if isinstance(max_dets_per_image, tuple):
        return list(max_dets_per_image)
    return max_dets_per_image
