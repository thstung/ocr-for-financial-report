from pdf2image import convert_from_path
import numpy as np
import cv2
from layout_detection import layout_detection 
from ocr_text import ocr_image
from table_detection import table_detect
from table_structure_rec import table_structure_recognize, get_cells
import os
import time
import csv
import random

start = time.time()

input = "../data/hsg.pdf"
output = '../output'

pages = convert_from_path(input, poppler_path="../poppler-0.68.0/bin")

file_name = os.path.basename(input)
image_name = os.path.splitext(file_name)[0]

for i, page in enumerate(pages):
    predict_layout = layout_detection(page)
    list_text = []
    image_page = np.array(page)
    num_table = 0
    for layout in predict_layout:
        if layout["label"] in ['text', 'title', 'list']:
            left = layout["bbox"][0]
            top = layout["bbox"][1]
            right = layout["bbox"][2]
            bot = layout["bbox"][3]
            image_page = cv2.rectangle(image_page, (left, top), (right, bot), color=(255,0,0), thickness=2)
            img_text = page.crop((left, top, right, bot))
            text = ocr_image(img_text)
            list_text.extend(text)
        elif layout["label"] == 'table':
            left = layout["bbox"][0]
            top = layout["bbox"][1]
            right = layout["bbox"][2]
            bot = layout["bbox"][3]
            img_table = page.crop((left, top, right, bot))
            # print(img_table.size)
            # r = random.randint(0, 255)
            # g = random.randint(0, 255)
            # b = random.randint(0, 255)
            image_page = cv2.rectangle(image_page, (left, top), (right, bot), color=(0,0,255), thickness=2)
            table_structure = table_structure_recognize(img_table)
            image_table = np.array(img_table)
            for bbox_struc in table_structure:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                image_table = cv2.rectangle(image_table, (int(bbox_struc['bbox'][0]),int(bbox_struc['bbox'][1])),
                                             (int(bbox_struc['bbox'][2]),int(bbox_struc['bbox'][3])), color=(r,g,b), thickness=2)
            cv2.imwrite(output + '/' + image_name + '_' + str(i) + '_' + str(num_table) + '.png',image_table)
            cells = get_cells(table_structure)
            with open(output + '/' + image_name + '_' + str(i) + '_' + str(num_table) + '.csv', 'a', newline='', encoding='utf8') as file:
                writer = csv.writer(file)
                for cells_row in cells:
                    list_text_row = []
                    for cell in cells_row:
                        img_cell = img_table.crop(cell)
                        text_of_cell = ocr_image(img_cell)
                        text_of_cell = ' '.join(text_of_cell)
                        list_text_row.append(text_of_cell)
                    
                    print(list_text_row)
                    writer.writerow(list_text_row)
    cv2.imwrite(output + '/' + image_name + '_' + str(i) + '.png',image_page)
    with open(output + '/' + image_name + '_' + str(i) + '.txt', 'w', encoding='utf-8') as f:
            for text in list_text:
                 f.write(text + '\n')

    print("Finish!!!!!!!!!!!!!!")
    
print("Time process:", time.time() - start)

