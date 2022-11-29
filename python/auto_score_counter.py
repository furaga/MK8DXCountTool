import re
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
from lib import digit_ocr


def parse_args():
    parser = argparse.ArgumentParser(description='Hand Tracking Demo')
    parser.add_argument('--img_path', type=Path, required=True)
    parser.add_argument('--ocr_path', type=Path, default="")
    parser.add_argument('--my_tag', type=str, default="RR")
    args = parser.parse_args()
    return args


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


def common_prefix(n1, n2):
    prefix = ""
    for c1, c2 in zip(n1, n2):
        if c1 != c2:
            break
        prefix += c1
    return prefix


def common_suffix(n1, n2):
    return common_prefix(n1[::-1], n2[::-1])[::-1]


def estimate_tags(names):
    tags = []
    for i, n in enumerate(names):
        # 既存のtagがあるか
        found = False
        for t, is_prefix in tags:
            if is_prefix and n.startswith(t):
                found = True
                break
            if not is_prefix and n.endswith(t):
                found = True
                break
        if found:
            continue

        new_tag = []
        for j, n2 in enumerate(names):
            if i != j:
                prefix = common_prefix(n, n2)
                suffix = common_suffix(n, n2)
                if len(prefix) > 0:
                    if len(new_tag) <= 0 or len(new_tag[0]) < len(prefix):
                        new_tag = prefix, True
                elif len(suffix) > 0:
                    if len(new_tag) <= 0 or len(new_tag[0]) < len(suffix):
                        new_tag = suffix, False

        if len(new_tag) <= 0:
            print(f"tag for {n} not found. Use its name as tag.")
            tags.append((n, True))
        else:
            tags.append(new_tag)

    return tags


def fallback_ocr_path(img_path):
    return img_path.parent.parent / "scores_ocr" / (img_path.stem + ".txt")


def load_ocr_results(ocr_path):
    with open(ocr_path, encoding="utf8") as f:
        texts = []
        bounding_vertexes = []
        for line in f:
            tokens = line.strip().split(',')
            tokens = [t.strip() for t in tokens]
            if len(tokens) < 9:
                continue
            text = tokens[0]
            pts = np.reshape(
                [int(t) for t in tokens[1:]], (-1, 2))
            texts.append(text)
            bounding_vertexes.append(pts)

    return texts, bounding_vertexes


def bounding_vertexes_to_boxes(bounding_vertexes):
    bounding_boxes = []
    for pts in bounding_vertexes:
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        bounding_boxes.append([x1, y1, x2 - x1, y2 - y1])
    return bounding_boxes[1:]


def merge_neighboring_bboxes(t_bounding_boxes):
    bounding_boxes = []
    for x, y, w, h in t_bounding_boxes:
        merged = False
        for i, b in enumerate(bounding_boxes):
            margin = 10
            l1 = b[0] - margin
            r1 = b[0] + b[2] + margin
            l2 = x - margin
            r2 = x + w + margin
            if abs(b[1] - y) < margin and l2 < r1 and l1 < r2:
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
    return bounding_boxes


def extract_regions(all_bounding_boxes, check_fn):
    boxes_ls = []
    for b in all_bounding_boxes:
        found = False
        for bs in boxes_ls:
            if check_fn(b, bs):
                bs.append(b)
                found = True
                break
        if not found:
            boxes_ls.append([b])

    best_idx = -1

    for i, bs in enumerate(boxes_ls):
        if len(bs) < 6:
            continue
        if best_idx < 0:
            best_idx = i
        else:
            # 左を優先する
            if boxes_ls[best_idx][0][0] > boxes_ls[i][0][0]:
                best_idx = i

    return boxes_ls[best_idx]


def extract_name_regions(all_bounding_boxes):
    return extract_regions(
        all_bounding_boxes,
        check_fn=lambda b, bs: abs(b[0] - bs[0][0]) < 10
    )


def extract_score_regions(all_bounding_boxes):
    return extract_regions(
        all_bounding_boxes,
        check_fn=lambda b, bs: abs(b[0] + b[2] - bs[0][0] - bs[0][2]) < 10
    )


def correct_regions(name_regions, score_regions):
    # y-sort
    name_regions = sorted(name_regions, key=lambda b: b[1])

    score_regions = sorted(score_regions, key=lambda b: b[1])
    score_minx = np.min([b[0] for b in score_regions])
    score_maxx = np.max([b[0] + b[2] for b in score_regions])

    name_mean_dy = np.mean([name_regions[i + 1][1] - name_regions[i][1]
                            for i in range(len(name_regions) - 1)])
    name_minx = np.min([b[0] for b in name_regions])
    name_maxx = np.max([b[0] + b[2] for b in name_regions])

    # TODO: nameの漏れがあった場合の補完
    new_name_regions = []
    for i, b in enumerate(name_regions[:-1]):
        dy = name_regions[i + 1][1] - name_regions[i][1]
        drow = round(dy / name_mean_dy)
        if drow > 1:
            for j in range(drow - 1):
                interp_y = name_regions[i][1] + int((j + 1) * dy / drow)
                interp_h = name_regions[i][3]
                new_name_regions.append(
                    [name_minx, interp_y, name_maxx - name_minx, interp_h])
        new_name_regions.append(b)

    new_name_regions.append(name_regions[-1])

    print("new_name_regions", len(new_name_regions), len(name_regions))

    # nameに対応するscoreを見つける
    new_score_regions = []
    for i, n_box in enumerate(new_name_regions):
        found = False
        for s_box in score_regions:
            # n_boxとs_boxが同じくらいのY座標・高さかつs_boxがn_boxより右にある
            if abs(n_box[1] - s_box[1]) < 10 and abs(n_box[3] - s_box[3]) < 10 and n_box[0] + n_box[2] < s_box[0]:
                new_score_regions.append(s_box)
                found = True
                break
        if not found:
            new_score_regions.append(None)

    # scoreの左端は下の項目以上であるはずなのでそれを保証する
    prev_x = 1e4
    for i in range(len(new_score_regions) - 1, -1, -1):
        b = new_score_regions[i]
        if b is None:
            continue
        new_x = min(prev_x, b[0])
        new_score_regions[i] = new_x, b[1], b[2] + b[0] - new_x, b[3]
        prev_x = new_x

    # 対応するscoreが見つからなかった場合のfallback
    for i in range(0, len(new_score_regions)):
        if new_score_regions[i] is None:
            # fallback
            _, y, _, h = new_name_regions[i]
            if i <= 0:
                x = score_minx
                w = score_maxx - score_minx
            else:
                x, _, w, _ = new_score_regions[i - 1]
            new_score_regions[i] = [x, y, w, h]

    return new_name_regions, new_score_regions


def find_texts(regions, all_texts, all_bounding_vertexes):
    texts = []
    for x, y, w, h in regions:
        x2 = x + w
        y2 = y + h

        contained = []
        for text, pts in zip(all_texts, all_bounding_vertexes):
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            if x <= min_pt[0] and y <= min_pt[1] and max_pt[0] <= x2 and max_pt[1] <= y2:
                contained.append((text, min_pt[0]))

        # x座標順に並べる
        contained = sorted(contained, key=lambda item: item[1])
        final_text = "".join([t for t, _ in contained])
        texts.append(final_text)

    return texts


def try_int(t):
    try:
        return True, int(t)
    except Exception:
        return False, 0


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return texts


def correct_names(names, name_regions, img_bgr):
    margin = 16
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for i in range(len(names)):
        if names[i] != "":
            continue
        x, y, w, h = name_regions[i]
        top = max(0, y-margin)
        bottom = min(y+h+margin, img_bgr.shape[0])
        left = max(0, x-margin)
        right = min(x+w+margin, img_bgr.shape[1])

        crop = img_bgr[top:bottom, left:right]

        cv2.imwrite("__tmp__.png", crop)
        texts = detect_text("__tmp__.png")
        new_name = ""
        print(texts)
        if len(texts) >= 1:
            t = texts[0].description.encode('cp932', "ignore")
            new_name = t.decode('cp932')

        print(
            f"name {i} is not valid. Use Google Vision API -> '{new_name}'")

        names[i] = new_name

    return names


def correct_scores(scores, score_regions, img_bgr):
    prev_score = 0
    for i in range(len(scores) - 1, 0, -1):
        ret, val = try_int(scores[i])
        if not ret:
            scores[i] = ""
        # スコアは降順なので、そうなってなければ認識ミスしている
        if val < prev_score:
            scores[i] = ""
        prev_score = val

    margin = 16
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for i in range(len(scores)):
        if scores[i] != "":
            continue
        x, y, w, h = score_regions[i]
        if w < h:
            w += h
            x -= h
        top = max(0, y-margin)
        bottom = min(y+h+margin, img_bgr.shape[0])
        left = max(0, x-margin)
        right = min(x+w+margin, img_bgr.shape[1])
        crop = img_gray[top:bottom, left:right]

        cv2.imwrite(
            f"data/digital_testdata/{args.img_path.stem}_{i}.png", crop)

        # デジタル数字はGoogle OCRで認識しづらい。がんばって自力で認識する
        ret, score = digit_ocr.detect_digit(crop)
        if not ret:
            _, score = digit_ocr.detect_digit(255 - crop)

        scores[i] = str(score)

    return scores


def main(args):
    ocr_path = args.ocr_path if len(
        str(args.ocr_path)) <= 0 else fallback_ocr_path(args.img_path)
    print(ocr_path)

    all_texts, all_bounding_vertexes = load_ocr_results(ocr_path)

    all_bounding_boxes = bounding_vertexes_to_boxes(all_bounding_vertexes)
    all_bounding_boxes = merge_neighboring_bboxes(all_bounding_boxes)

    name_regions = extract_name_regions(all_bounding_boxes)
    score_regions = extract_score_regions(all_bounding_boxes)

    name_regions, score_regions = correct_regions(name_regions, score_regions)

    names = find_texts(name_regions, all_texts, all_bounding_vertexes)
    scores = find_texts(score_regions, all_texts, all_bounding_vertexes)
    print(names)
    print(scores)

    img_bgr = cv2.imread(str(args.img_path))
    names = correct_names(names, name_regions, img_bgr)
    scores = correct_scores(scores, score_regions, img_bgr)

    for i, (n, s) in enumerate(zip(names, scores)):
        print(f"{i+1}: {n}, {s}")

    # タグを推定
    tags = estimate_tags(names)

    def has_tag(n, t):
        if t[1] and n.startswith(t[0]):
            return True
        if not t[1] and n.endswith(t[0]):
            return True
        return False

    def to_int(s):
        try:
            return int(s)
        except Exception:
            return 0

    tag_scores = {t[0]: np.sum([to_int(s) for n, s in zip(
        names, scores) if has_tag(n, t)]) for t in tags}

    # my_tagとの差分付きでスコア大きい順に表示
    my_tag = [t for t, _ in tags if args.my_tag in t]
    my_tag = my_tag[0] if len(my_tag) >= 1 else tags[0][0]
    sorted_tag_scores = sorted(tag_scores.items(), key=lambda ts: -ts[1])
    print("====================")
    for tag, score in sorted_tag_scores:
        diff = score - tag_scores[my_tag]
        print(f"{tag}: {score} ({'+' if diff >= 0 else ''}{diff})")
    print("====================")

    for x, y, w, h in all_bounding_boxes:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2, -1)
    for x, y, w, h in name_regions:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 255), 4, -1)
    for x, y, w, h in score_regions:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 4, -1)

    cv2.imshow("img_bgr", img_bgr)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)
