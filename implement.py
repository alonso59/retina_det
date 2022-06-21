# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
import cv2
from pathlib import Path

import torch
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


@torch.no_grad()
def impl(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=None,  # array
        imgsz=(224, 224),  # inference size (height, width)
        device='cuda',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
):
    im = cv2.resize(source, (source.shape[0], source.shape[1]))
    im0 = im.copy()
    im = cv2.resize(im, imgsz)
    im = np.array(im).transpose((2, 0, 1)) / 255.
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    im = torch.from_numpy(im).float().to(device)

    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # Inference
    pred = model(im, augment=False, visualize=False)
    # NMS
    
    # im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    pred = non_max_suppression(pred)
    macula = 0
    od = 0
    for i, det in enumerate(pred):

        annotator = Annotator(im0, line_width=2, example=str(names))
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            if cls == 1:
                od = conf.item()
            if cls == 0:
                macula = conf.item()
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    # Stream results
    im0 = annotator.result()

    return im0, macula, od
