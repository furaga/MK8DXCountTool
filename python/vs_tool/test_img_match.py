import cv2
import numpy as np

MIN_ROD_CONTRAST = 32


def is_digit_or_segment(roi):
    top, left, right, bottom, area = roi
    height, width = bottom + 1 - top, right + 1 - left
    h_per_w = height / width
    if area / (height * width) > .99 or area / (height * width) < .1:
        return False
    if area / (height * width) > .85 and .5 <= h_per_w <= 2:
        return False
    if h_per_w < .3 or h_per_w > 10:
        return False
    return True


def rod_thresholding(roi, image_height, image_width, min_area_ratio=.0025, max_area_ratio=.2,
                     min_height_per_width=.5, max_height_per_width=10,
                     min_height_ratio=.15):
    top, left, right, bottom, area = roi
    image_area = image_height * image_width
    if (bottom + 1 - top) / image_height <= min_height_ratio:
        if area / image_area <= min_area_ratio:
            return False
        if area / image_area >= max_area_ratio:
            return False
    h_per_w = (bottom + 1 - top) / (right + 1 - left)
    if h_per_w <= min_height_per_width:
        return False
    if h_per_w >= max_height_per_width:
        return False
    if h_per_w <= 4 and area / ((bottom + 1 - top) * (right + 1 - left)) > .85:
        return False
    return True


SEGMENTS2DIGIT = {0b0000000: -1, 0b0000001: -1, 0b0000010: 1, 0b0000011: -1, 0b0000100: -1,
                  0b0000101: -1, 0b0000110: -1, 0b0000111: -1, 0b0001000: -1, 0b0001001: -1,
                  0b0001010: -1, 0b0001011: -1, 0b0001100: -1, 0b0001101: -1, 0b0001110: -1,
                  0b0001111: -1, 0b0010000: 1, 0b0010001: -1, 0b0010010: 1, 0b0010011: -1,
                  0b0010100: -1, 0b0010101: -1, 0b0010110: -1, 0b0010111: -1, 0b0011000: -1,
                  0b0011001: -1, 0b0011010: 4, 0b0011011: 3, 0b0011100: -1, 0b0011101: 2,
                  0b0011110: -1, 0b0011111: -1, 0b0100000: -1, 0b0100001: -1, 0b0100010: -1,
                  0b0100011: -1, 0b0100100: -1, 0b0100101: -1, 0b0100110: -1, 0b0100111: -1,
                  0b0101000: -1, 0b0101001: -1, 0b0101010: 4, 0b0101011: 5, 0b0101100: -1,
                  0b0101101: -1, 0b0101110: -1, 0b0101111: 6, 0b0110000: -1, 0b0110001: -1,
                  0b0110010: 4, 0b0110011: -1, 0b0110100: -1, 0b0110101: -1, 0b0110110: -1,
                  0b0110111: 0, 0b0111000: 4, 0b0111001: -1, 0b0111010: 4, 0b0111011: 9,
                  0b0111100: -1, 0b0111101: -1, 0b0111110: -1, 0b0111111: 8, 0b1000000: -1,
                  0b1000001: -1, 0b1000010: -1, 0b1000011: -1, 0b1000100: -1, 0b1000101: -1,
                  0b1000110: -1, 0b1000111: -1, 0b1001000: -1, 0b1001001: -1, 0b1001010: -1,
                  0b1001011: 3, 0b1001100: -1, 0b1001101: -1, 0b1001110: -1, 0b1001111: 6,
                  0b1010000: -1, 0b1010001: -1, 0b1010010: 7, 0b1010011: 3, 0b1010100: -1,
                  0b1010101: 2, 0b1010110: -1, 0b1010111: 0, 0b1011000: -1, 0b1011001: 2,
                  0b1011010: 3, 0b1011011: 3, 0b1011100: 2, 0b1011101: 2, 0b1011110: -1,
                  0b1011111: 8, 0b1100000: -1, 0b1100001: -1, 0b1100010: 7, 0b1100011: 5,
                  0b1100100: -1, 0b1100101: -1, 0b1100110: -1, 0b1100111: 0, 0b1101000: -1,
                  0b1101001: 5, 0b1101010: 5, 0b1101011: 5, 0b1101100: -1, 0b1101101: 6,
                  0b1101110: 6, 0b1101111: 6, 0b1110000: 7, 0b1110001: -1, 0b1110010: 7,
                  0b1110011: 0, 0b1110100: -1, 0b1110101: 0, 0b1110110: 0, 0b1110111: 0,
                  0b1111000: -1, 0b1111001: 9, 0b1111010: 9, 0b1111011: 9, 0b1111100: -1,
                  0b1111101: 8, 0b1111110: 8, 0b1111111: 8}


def ison(roi_bin, is_vertical: bool, ret_val: int) -> int:
    if is_vertical:
        thresh = roi_bin.shape[0] * .6
        if (np.count_nonzero(roi_bin, axis=0) > thresh).any():
            return ret_val
    else:
        thresh = roi_bin.shape[1] * .7
        if (np.count_nonzero(roi_bin, axis=1) > thresh).any():
            return ret_val
    return 0


def rod2digit(rod, rod_bin):
    height_rod, width_rod = rod_bin.shape
    _, labels, stats = cv2.connectedComponentsWithStats(rod_bin)[:3]
    # Remove tiny blobs.
    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]
        if max(w, h) < min(width_rod, height_rod) * .15:
            rod_bin[i == labels] = 0

    if height_rod / width_rod > 3:
        read_digit = 1
    else:
        vline1, vline2 = width_rod // 3, width_rod * 2 // 3
        hline1, hline2, hline3 = height_rod // 3, height_rod // 2, height_rod * 2 // 3
        segments = [rod_bin[:hline1, vline1:vline2], rod_bin[:hline2, :vline1], rod_bin[:hline2, vline2:],
                    rod_bin[hline1:hline3, vline1:vline2], rod_bin[hline2:,
                                                                   :vline1], rod_bin[hline2:, vline2:],
                    rod_bin[hline3:, vline1:vline2]]
        is_verticals = [False, True, True, False, True, True, False]
        ret_vals = [64, 32, 16, 8, 4, 2, 1]
        read_digit = SEGMENTS2DIGIT[sum(
            list(map(ison, segments, is_verticals, ret_vals)))]
    if -1 == read_digit and np.max(np.count_nonzero(rod_bin, axis=0)) / max(1, np.max(np.count_nonzero(rod_bin, axis=1))) > 3.5:
        read_digit = 1
    return {'digit': int(read_digit), 'left': int(rod[1]), 'top': int(rod[0]),
            'width': int(width_rod), 'height': int(height_rod)}


def crop_digits(roi_gray):
    KERNEL_4N = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    roi_gray = roi_gray.copy()
    roi_gray_enhanced = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
    height, width = roi_gray_enhanced.shape
    EDGELEN_PERCENT = (height + width) / 200
    EDGELEN_PERCENT_LONG = max(height, width) / 100

    rods = []  # ROD = Region of a Digit
    rods_img = np.zeros([height, width], np.uint8)

    # Remove shadows near the display contour.
    inner_x1, inner_x2 = int(width * .02), int(width * .98)
    inner_y1, inner_y2 = int(height * .05), int(height * .95)
    roi_bin_inner = cv2.adaptiveThreshold(roi_gray_enhanced[inner_y1:inner_y2, inner_x1:inner_x2], 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                          max(int(min(height, width) / 2 * .5), 1) * 2 + 1, 2)
    digits_brightness = np.mean(
        roi_gray_enhanced[inner_y1:inner_y2, inner_x1:inner_x2][roi_bin_inner > 127])
    roi_bin = cv2.threshold(roi_gray_enhanced, 255,
                            digits_brightness, cv2.THRESH_BINARY)[1]
    roi_bin[inner_y1:inner_y2, inner_x1:inner_x2] = roi_bin_inner.copy()

    tmp = cv2.erode(roi_bin.copy(), KERNEL_4N,
                    iterations=int(EDGELEN_PERCENT / 4 * .5) + 1)
    _, labels, stats = cv2.connectedComponentsWithStats(tmp)[:3]
    for i, stat in enumerate(stats):
        if 0 == min(stat[0], stat[1]) or width == stat[0] + stat[2] or height == stat[1] + stat[3]:
            roi_bin[i == labels] = 0

    # Cut off bright area near the display contour and long bright line.
    tmp = roi_bin.copy()
    for i, bright in enumerate(np.count_nonzero(roi_bin, axis=1)):
        if bright < width * .8:
            continue
        tmp[i, :] = 0
    for i, bright in enumerate(np.count_nonzero(roi_bin, axis=0)):
        if bright < height * .95:
            continue
        tmp[:, i] = 0
    roi_bin = tmp.copy()

    # Remove blobs which is not a segment or a digit.
    labels, stats = cv2.connectedComponentsWithStats(roi_bin)[1:3]
    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]
        if roi_bin[i == labels][0] < 128:
            continue
        right, bottom = left + w, top + h
        if not is_digit_or_segment((top, left, right, bottom, stat[4])):
            roi_bin[i == labels] = 0

    # Remove thin lines.

    kernel = np.ones((5, 5), np.uint8)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel)
    # roi_bin = cv2.erode(roi_bin, KERNEL_4N, iterations=int(EDGELEN_PERCENT / 2 * .5) + 1)
    # roi_bin = cv2.dilate(roi_bin, KERNEL_4N, iterations=int(EDGELEN_PERCENT / 2 * .5) + 1)

    n_labels, labels, stats = cv2.connectedComponentsWithStats(roi_bin)[:3]
    json_digits = {'digits': []}

    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]

        if h < height * 0.5 or height * 0.99 < h:
            continue

        right, bottom = left + w - 1, top + h - 1
        if not rod_thresholding((top, left, right, bottom, stat[4]), height, width, max_area_ratio=.4):
            continue

        # Remove low contrast area. (ex. shadow but not a digit)
        rod_gray = roi_gray[top:bottom+1, left:right+1].copy()
        height_rod, width_rod = rod_gray.shape
        nth = width_rod * \
            height_rod // (40 if height_rod / width_rod > 3 else 12)
        min_value = np.partition(rod_gray.ravel(), nth)[nth]
        rod_gray[(i != labels)[top:bottom+1, left:right+1]] = min_value
        if np.max(rod_gray) - min_value < MIN_ROD_CONTRAST:
            continue

        rods.append((left, top, left + w, top + h, None))

    for rod in rods:
        left, top, right, bottom, _ = rod
        cr = roi_bin[top:bottom, left:right]
        print(rod2digit(rod, cr))

    roi_bgr = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
    for i, (left, top, right, bottom, _) in enumerate(rods):
        cr = roi_gray_enhanced[top:bottom, left:right]
        cv2.imshow(f"crop{i}.png", cr)
        cv2.rectangle(roi_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow("roi_bgr", roi_bgr)
    cv2.waitKey(0)


img1 = cv2.imread("data/scores_crop/11.png")
img2 = cv2.imread("data/digital_numbers/1.png")


img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

crop_digits(img_gray)
exit()

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(img_gray)

# template = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# w, h = template.shape[::-1]

# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.5
# loc = np.where( res >= np.max(res))
# print(loc)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imshow('res.png', img1)
cv2.waitKey(0)
cv2.imshow('cl1.png', cl1)
cv2.waitKey(0)

# # OBR 特徴量検出器を作成する。
# detector = cv2.ORB_create()

# # 特徴点を検出する。
# kp1, desc1 = detector.detectAndCompute(img1, None)
# kp2, desc2 = detector.detectAndCompute(img2, None)

# # マッチング器を作成する。
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# print(desc1)
# print(desc2)

# # マッチングを行う。
# matches = bf.knnMatch(desc1, desc2, k=2)

# # レシオテストを行う。
# good_matches = []
# thresh = 0.7
# for first, second in matches:
#     if first.distance < second.distance * thresh:
#         good_matches.append(first)

# # マッチング結果を描画する。
# dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
# cv2.imshow(dst)
# cv2.waitKey(0)
