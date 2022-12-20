import argparse
import cv2
import numpy as np
from glob import glob
import os
import sys
from pathlib import Path
from playsound import playsound

def parse_args():
    parser = argparse.ArgumentParser(description='Hand Tracking Demo')
    parser.add_argument('--template_dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.98)
    parser.add_argument('--threshold_banana', type=float, default=0.95)
    args = parser.parse_args()
    return args

def load_templates(template_dir):
    templates = {}
    img_paths = list(template_dir.glob("*.png"))
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        templates[img_path.stem] = img
    return templates


def main(args):
    templates = load_templates(args.template_dir)

    cap = cv2.VideoCapture(3)
    i_frame = 0
    prev_item = ""
    while True:
        ret, img = cap.read()
        if not ret:
            break

        cropped = img[:img.shape[0]//2, :img.shape[1]//2] # TODO
        for name, tmpl in templates.items():
            result = cv2.matchTemplate(cropped, tmpl, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            thr = args.threshold if "bana" not in name else args.threshold_banana
            if max_val > args.threshold:
                top_left = max_loc
                h, w = tmpl.shape[:2]
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 4)
                if name != prev_item:
                    if '1' in name and '3' in prev_item and name[:-1] == prev_item[:-1]:
                        pass
                    else:
                        print("Found", name, "prob=", max_val)
                        playsound(f"data/voice/{name}.mp3", block = False)
                        prev_item = name

        cv2.imshow("img", img)
        if ord('s') == cv2.waitKey(1):
            cv2.imwrite(f"{i_frame:03d}.png", img)

        i_frame += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)

