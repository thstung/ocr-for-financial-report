from PIL import Image as image_main
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'

vietocr = Predictor(config)

paddle_ocr = PaddleOCR()

def ocr_image(image):
    image_array = np.array(image)
    result = paddle_ocr.ocr(image_array, rec=False)

    list_text = []

    for line in result:
        id = 0
        for bbox in line:
            left = bbox[0][0]
            top = bbox[0][1]
            right = bbox[2][0]
            bot = bbox[2][1]
            im_crop = image.crop((left, top, right, bot))

            recognized_text = vietocr.predict(im_crop)
            list_text.insert(0, recognized_text)
    return list_text

if __name__ == "__main__":
    image_path = "../fpt/detected_pages/fpt_page_0_text.jpg"
    image_pil = image_main.open(image_path)
    detection = ocr_image(image_pil)
    print(detection)