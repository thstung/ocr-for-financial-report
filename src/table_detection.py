import json
import sys

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
from settings import TD_MODEL, TD_MODEL_CONFIG
import random

sys.path.append("../detr")
from models import build_model

class_map = {'table': 0, 'table rotated': 1, 'no object': 2}

detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}

det_class_idx2name = {v:k for k, v in class_map.items()}

det_model_path = "../model/td/pubtables1m_detection_detr_r18.pth"
det_config_path = TD_MODEL_CONFIG
with open(det_config_path, 'r') as f:
    det_config = json.load(f)
det_args = type('Args', (object,), det_config)
det_args.device = 'cpu'
det_model, _, _ = build_model(det_args)
print("Table Detection model initialized.")

def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]
    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            if bbox == None:
              continue
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def objects_to_crops(img, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)

        cropped_table['image'] = cropped_img

        table_crops.append(cropped_table)

    return table_crops

def table_detect(img, out_objects=True, out_crops=False, crop_padding=10):
        out_formats = {}
        if det_model is None:
            print("No detection model loaded.")
            return out_formats

        img = Image.open(img)

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = det_model([img_tensor.to('cpu')])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, det_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not out_crops:
            return out_formats

        # Crop image and tokens for detected table
        if out_crops:
            tables_crops = objects_to_crops(img, objects, detection_class_thresholds,
                                            padding=crop_padding)
            out_formats['crops'] = tables_crops

        return out_formats

if __name__ == "__main__":
    import cv2
    output = table_detect("../data/tsr/images/hsg_9.jpg")
    image = cv2.imread("../data/tsr/images/hsg_9.jpg")
    for i in output['objects']:
        # print(i['bbox'][0])
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        image = cv2.rectangle(image, (int(i['bbox'][0]),int(i['bbox'][1])), (int(i['bbox'][2]),int(i['bbox'][3])), color=(r,g,b), thickness=2)
        cv2.imwrite("img.jpg", image)
    print(output)