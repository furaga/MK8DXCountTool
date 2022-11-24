import argparse
import cv2
import numpy as np
from glob import glob
import os
import sys
from pathlib import Path
from playsound import playsound
import pyocr
from PIL import Image, ImageEnhance

def parse_args():
    parser = argparse.ArgumentParser(description='Hand Tracking Demo')
    parser.add_argument('--img_path', type=Path, required=True)
    args = parser.parse_args()
    return args

import re

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image    


def replace_chars(text):
    """
    Replaces all characters instead of numbers from 'text'.
    
    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'\d+', text)
    result_number = ''.join(list_of_numbers)
    return result_number


def main(args):
    img = cv2.imread(str(args.img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thr = cv2.threshold(img,175,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3,3),np.uint8)
    closed = cv2.dilate(closed, kernel, iterations = 1)
    
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    t_bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = []
    for x, y, w, h in t_bounding_boxes:
        if 100 < x  and x + w < 360 and 10 < h < 40:
            merged = False
            for i, b in enumerate(bounding_boxes):
                if abs(b[1] - y) < 5:
                    bounding_boxes[i] = [
                        min(bounding_boxes[i][0], x),
                        min(bounding_boxes[i][1], y),
                        max(bounding_boxes[i][0] + bounding_boxes[i][2], x + w),
                        max(bounding_boxes[i][1] + bounding_boxes[i][3], y + h),
                    ]
                    bounding_boxes[i][2] -= bounding_boxes[i][0]
                    bounding_boxes[i][3] -= bounding_boxes[i][1]
                    merged = True
                    break
            if not merged:
                bounding_boxes.append((x, y, w, h))

    closed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(closed, (x, y), (x + w, y + h), (0, 0, 255), 2, -1)

    # cv2.imshow("img", img)
    # cv2.imshow("closed", closed)
    
    # cv2.waitKey(0)

    
    #OCRエンジン取得
    tools = pyocr.get_available_tools()
    tool = tools[0]
    builder = pyocr.builders.TextBuilder(tesseract_layout=6)

    import pytesseract
    img_pil = cv2pil(thr)  #  Image.open(args.img_path)
    # img_g = img_pil
    # enhancer= ImageEnhance.Contrast(img_g) #コントラストを上げる
    # img_pil = enhancer.enhance(2.0) #コントラストを上げる
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        margin = 3
        crop = img_pil.crop([x - margin, y - margin, x + w + margin, y + h + margin])
        crop.save(f"{i:03d}.png")
        txt_pyocr = tool.image_to_string(crop , lang='jpn+eng', builder=builder)
        print(txt_pyocr)

        crop = img_pil.crop([img_pil.width - 40, y - margin - 5, img_pil.width, y + h + margin + 5])
        crop.save(f"s{i:03d}.png")
        txt_pyocr = tool.image_to_string(crop , lang='ssd', builder=builder)
        print(txt_pyocr)


if __name__ == '__main__':
    args = parse_args()
    main(args)

