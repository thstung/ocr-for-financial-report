import json
import sys

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
from settings import TSR_MODEL, TSR_MODEL_CONFIG

sys.path.append("../detr")
from models import build_model

str_model_path = TSR_MODEL
str_config_path = TSR_MODEL_CONFIG
with open(str_config_path, 'r') as f:
    str_config = json.load(f)
str_args = type('Args', (object,), str_config)
str_args.device = 'cpu'
str_model, _, _ = build_model(str_args)
print("Structure model initialized.")

str_model.load_state_dict(torch.load(str_model_path,
                                      map_location=torch.device('cpu')), strict=False)
str_model.to('cpu')
str_model.eval()

str_class_name2idx = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
str_class_idx2name = {v:k for k, v in str_class_name2idx.items()}

str_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 0.5
}

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        print(width, height)
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
                            'bbox': [int(elem) for elem in bbox]})

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


def table_structure_recognize(img, out_objects=False, out_cells=False,
                  out_html=False, out_csv=False):
    out_formats = {}
    if str_model is None:
        print("No structure model loaded.")
        return out_formats
    print("Image", img)
    # Transform the image how the model expects it
    img_tensor = structure_transform(img)

    # Run input image through the model
    outputs = str_model([img_tensor.to("cpu")])
    # Post-process detected objects, assign class labels
    objects = outputs_to_objects(outputs, img.size, str_class_idx2name)
    return objects

def get_cells(object_table):
    table_columns = sorted([obj for obj in object_table if obj['label'] in ['table column', 'table column header']], key=lambda obj: obj['bbox'][0], reverse=False)
    table_rows = sorted([obj for obj in object_table if obj['label'] in ['table row', 'table projected row header']], key=lambda obj: obj['bbox'][1], reverse=False)
    # print(table_columns)
    cells = []
    for row_num, row in enumerate(table_rows):
        cells_row = []
        for column_num, column in enumerate(table_columns):
            cells_row.append([column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]])
        cells.append(cells_row)
    return cells

if __name__ == "__main__":
    output = table_structure_recognize("../data/tsr/hsg_20.jpg")
    print(output)