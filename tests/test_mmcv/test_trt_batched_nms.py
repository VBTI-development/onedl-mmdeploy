# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.mmcv.ops import TRTBatchedNMSop


def test_trt_batched_nms_basic():
    boxes = torch.tensor([[[10.0, 10.0, 20.0, 20.0], [15.0, 15.0, 25.0, 25.0],
                           [30.0, 30.0, 40.0, 40.0]]])  # [1, 3, 4]
    scores = torch.tensor([[0.9, 0.8,
                            0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]).unsqueeze(
                                0)  # [1, 3, 3] (batch, num_boxes, num_classes)
    num_classes = 3
    pre_topk = 3
    after_topk = 2
    iou_threshold = 0.5
    score_threshold = 0.0
    background_label_id = -1
    return_index = True

    dets, labels, indices = TRTBatchedNMSop.apply(
        boxes, scores, num_classes, pre_topk, after_topk, iou_threshold,
        score_threshold, background_label_id, return_index)
    assert dets.shape[0] == 1
    assert dets.shape[2] == 5
    assert labels.shape[0] == 1
    assert indices.shape[0] == 1
