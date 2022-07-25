#!/usr/bin/env python3
import torch
from iou import iou

def nms(predictions, iou_threshold, prob_threshold, format="corner"):
    # predictions = [[1, 0.9, x1, y1, x2, y2]]

    assert type(predictions) == list

    bboxes = [box for box in predictions if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nms_bboxes = []

    while bboxes:
        current_box = bboxes.pop(0)

        bboxes = [
             box for box in bboxes
             if box[0] != current_box[0]
             or iou(torch.tensor(current_box[2:]),
                   torch.tensor(box[2:]),
                   format=format)
             < iou_threshold
        ]
        nms_bboxes.append(current_box)

    return nms_bboxes
