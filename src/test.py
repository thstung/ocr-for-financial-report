from pdf2image import convert_from_path
import numpy as np
from PIL import Image
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

input = "../company_img"
output = '../company_img_output'

files = os.listdir(input)

# Duyệt qua từng file và đọc nội dung
for file_name in files:
    file_path = os.path.join(input, file_name) 
    image_name = os.path.splitext(file_name)[0]
    img_table = Image.open(file_path).convert('RGB')
    image_table = np.array(img_table)
    table_structure = table_structure_recognize(img_table)
    
    for bbox_struc in table_structure:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        image_table = cv2.rectangle(image_table, (int(bbox_struc['bbox'][0]),int(bbox_struc['bbox'][1])),
                                        (int(bbox_struc['bbox'][2]),int(bbox_struc['bbox'][3])), color=(r,g,b), thickness=2)
    cv2.imwrite(output + '/' + image_name + '.png',image_table)
    cells = get_cells(table_structure)
    with open(output + '/' + image_name + '.csv', 'a', newline='', encoding='utf8') as file:
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

    print("Finish!!!!!!!!!!!!!!")
    
print("Time process:", time.time() - start)

