import cv2
import numpy
from PIL import Image as image_main, ImageDraw
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import time
from settings import LAYOUT_DETECTION_MODEL, LAYOUT_DETECTION_MODEL_CONFIG


# Model, threshold score, class labels, and example image - be sure to replace with your own
# image_path = '../Multi_Type_TD_TSR/images/color_invariance_example.jpg'
model_path = LAYOUT_DETECTION_MODEL
model_zoo_config_name = LAYOUT_DETECTION_MODEL_CONFIG
prediction_score_threshold = 0.7
class_labels = ['text', 'title', 'list', 'table', 'figure']

# Detectron config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

# Detectron predictor
predictor = DefaultPredictor(cfg)

def layout_detection(img):
    image_cv = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    outputs = predictor(image_cv)

    # Debug outputs
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    
    predict = []

    for i in range(0, len(pred_boxes)):
        bbox = pred_boxes[i].tensor.numpy()[0]
        print(bbox)
        score = round(float(scores[i].numpy()), 4)
        label_key = int(pred_classes[i].numpy())
        label = class_labels[label_key]
        predict.append({'bbox': [int(elem) for elem in bbox],
                         "label": label, "score": score})
    
    return predict

if __name__ == "__main__":
    image_path = "../fpt/detected_pages/fpt_page_0_text.jpg"
    image_pil = image_main.open(image_path)
    detection = layout_detection(image_pil)
    print(detection)
