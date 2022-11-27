import cv2
import numpy as np

MIN_ROD_CONTRAST = 32

def is_digit_or_segment(roi):
    top, left, right, bottom, area = roi
    height, width = bottom + 1 - top, right + 1 - left
    h_per_w = height / width
    if area / (height * width) > .99 or area / (height * width) < .1: return False
    if area / (height * width) > .85 and .5 <= h_per_w <= 2: return False
    if h_per_w < .3 or h_per_w > 10: return False
    return True

def rod_thresholding(roi, image_height, image_width, min_area_ratio=.0025, max_area_ratio=.2,
                     min_height_per_width=.5, max_height_per_width=10,
                     min_height_ratio=.15):
    top, left, right, bottom, area = roi
    image_area = image_height * image_width
    if (bottom + 1 - top) / image_height <= min_height_ratio:
        if area / image_area <= min_area_ratio: return False
        if area / image_area >= max_area_ratio: return False
    h_per_w = (bottom + 1 - top) / (right + 1 - left)
    if h_per_w <= min_height_per_width: return False
    if h_per_w >= max_height_per_width: return False
    if h_per_w <= 4 and area / ((bottom + 1 - top) * (right + 1 - left)) > .85: return False
    return True

def crop_digits(roi_gray):
    KERNEL_4N = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    roi_gray = roi_gray.copy()
    roi_gray_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
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
    digits_brightness = np.mean(roi_gray_enhanced[inner_y1:inner_y2, inner_x1:inner_x2][roi_bin_inner > 127])
    roi_bin = cv2.threshold(roi_gray_enhanced, 255, digits_brightness, cv2.THRESH_BINARY)[1]
    roi_bin[inner_y1:inner_y2, inner_x1:inner_x2] = roi_bin_inner.copy()

    tmp = cv2.erode(roi_bin.copy(), KERNEL_4N, iterations=int(EDGELEN_PERCENT / 4 * .5) + 1)
    _, labels, stats = cv2.connectedComponentsWithStats(tmp)[:3]
    for i, stat in enumerate(stats):
        if 0 == min(stat[0], stat[1]) or width == stat[0] + stat[2] or height == stat[1] + stat[3]:
            roi_bin[i == labels] = 0

    # Cut off bright area near the display contour and long bright line.
    tmp = roi_bin.copy()
    for i, bright in enumerate(np.count_nonzero(roi_bin, axis=1)):
        if bright < width * .8: continue
        tmp[i, :] = 0
    for i, bright in enumerate(np.count_nonzero(roi_bin, axis=0)):
        if bright < height * .95: continue
        tmp[:, i] = 0
    roi_bin = tmp.copy()

    # Remove blobs which is not a segment or a digit.
    labels, stats = cv2.connectedComponentsWithStats(roi_bin)[1:3]
    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]
        if roi_bin[i == labels][0] < 128: continue
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
        nth = width_rod * height_rod // (40 if height_rod / width_rod > 3 else 12)
        min_value = np.partition(rod_gray.ravel(), nth)[nth]
        rod_gray[(i != labels)[top:bottom+1, left:right+1]] = min_value
        if np.max(rod_gray) - min_value < MIN_ROD_CONTRAST: 
            continue

        rods.append((left, top, left + w, top + h, None))

    roi_bgr = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
    for top, left, right, bottom, _ in rods:
        cv2.rectangle(roi_bgr, (top, left), (right, bottom), (0, 0, 255), 2)
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

cv2.imshow('res.png',img1)
cv2.waitKey(0)
cv2.imshow('cl1.png',cl1)
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