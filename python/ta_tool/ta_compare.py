import argparse
import pandas as pd
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Google OCR')
    # parser.add_argument('--img_dir', type=Path, default="data/scores")
    parser.add_argument('--img_dir1', type=Path, default="tick/best")
    parser.add_argument('--img_dir2', type=Path, default="tick/ref")
    parser.add_argument('--out_dir', type=Path, default="tick")
    args = parser.parse_args()
    return args


def get_time(img_path):
    ms = img_path.stem.split('@')[-1]
    ms = int(ms[:-2])
    return ms


def imread_safe(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def main(args):
    img_paths1 = sorted(list(args.img_dir1.glob("*.png")),
                        key=lambda p: get_time(p))
    img_paths2 = sorted(list(args.img_dir2.glob("*.png")),
                        key=lambda p: get_time(p))

    # 画像並べる
    render_img_size = 640
    margin = 20

    def _resize(img):
        h, w = img.shape[:2]
        ratio = min(render_img_size / w, render_img_size / h)
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        return img

    def draw_text(img, text, center):
        fontface = cv2.FONT_HERSHEY_SIMPLEX  # フォントの種類
        fontscale = 1.0  # 文字のスケール
        thickness = 2  # 文字の太さ
        (w, h), baseline = cv2.getTextSize(
            text, fontface, fontscale, thickness)
        cx, cy = center
        x = cx - w // 2
        y = cy + h // 2 - baseline // 2
        cv2.putText(img, text, (x, y), fontface,
                    fontscale, (0, 0, 0), thickness)

    for i in range(len(img_paths1) - 1):
        img1a = _resize(imread_safe(str(img_paths1[i])))
        img1b = _resize(imread_safe(str(img_paths1[i + 1])))
        img2a = _resize(imread_safe(str(img_paths2[i])))
        img2b = _resize(imread_safe(str(img_paths2[i + 1])))
        dt1 = get_time(img_paths1[i+1]) - get_time(img_paths1[i])
        dt2 = get_time(img_paths2[i+1]) - get_time(img_paths2[i])

        h, w, c = img1a.shape
        out_img = np.zeros((h * 2 + 60, w * 2 + margin, c), np.uint8)
        out_img.fill(255)
        out_img[:h, :w] = img1a
        out_img[h:2*h, :w] = img1b
        out_img[:h, -w:] = img2a
        out_img[h:2*h, -w:] = img2b

        if dt1 > dt2:
            draw_text(
                out_img, f"{args.img_dir1.stem} {dt1 / 1000:.3f}sec (+{(dt1-dt2) / 1000:.3f}sec)", (w // 2, h * 2 + 30))
        else:
            draw_text(
                out_img, f"{args.img_dir1.stem} {dt1 / 1000:.3f}sec ({(dt1-dt2) / 1000:.3f}sec)", (w // 2, h * 2 + 30))
        draw_text(out_img, f"{args.img_dir2.stem} {dt2 / 1000:.3f}sec",
                  (out_img.shape[1] - w // 2, h * 2 + 30))
        cv2.imwrite(str(args.out_dir / f"section{i:02d}.jpg"), out_img)

    rows = []
    start_ms1 = get_time(img_paths1[0])
    start_ms2 = get_time(img_paths2[0])
    prev_ms1, prev_ms2 = -1, -1
    for i, (img_path1, img_path2) in enumerate(zip(img_paths1, img_paths2)):
        ms1 = get_time(img_path1)
        ms2 = get_time(img_path2)
        ms1 -= start_ms1
        ms2 -= start_ms2
        if prev_ms1 < 0:
            prev_ms1 = ms1
        if prev_ms2 < 0:
            prev_ms2 = ms2
        dt1 = (ms1 - prev_ms1) / 1000
        dt2 = (ms2 - prev_ms2) / 1000
        rows.append([
            i,
            ms1 / 1000,
            ms2 / 1000,
            (ms1 - ms2) / 1000,
            dt1,
            dt2,
            dt1 - dt2,
        ])
        prev_ms1 = ms1
        prev_ms2 = ms2

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "compare.csv", index=None, header=[
        "section",
        args.img_dir1.stem,
        args.img_dir2.stem,
        "Diff",
        args.img_dir1.stem + ": dt",
        args.img_dir2.stem + ": dt",
        "Diff",
    ])


if __name__ == '__main__':
    args = parse_args()
    main(args)
