import cv2
import imutils
import numpy as np


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def remove_glare(image1: np.ndarray, image2: np.ndarray, min_glare=215) -> tuple[np.ndarray, np.ndarray]:
    """
    This function accepts a pair of images in grayscale and removes over-exposed pixels from
    the images such as glare from sunlight by using min_glare as the minimum threshold.
    Over-exposed pixel values are assigned a value of 0 and resulting images are returned

    min_glare ranges from 0 to 255, with default being 215.
    """
    if -1 < min_glare < 256:
        mask = cv2.threshold(image1, min_glare, 255, cv2.THRESH_BINARY_INV)[1]
        masked_image1 = cv2.bitwise_and(image1, mask)
        del mask

        mask = cv2.threshold(image2, min_glare, 255, cv2.THRESH_BINARY_INV)[1]
        masked_image2 = cv2.bitwise_and(image2, mask)

        return masked_image1, masked_image2
    else:
        return image1, image2


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame1, frame2 = remove_glare(prev_frame.copy(), next_frame.copy())

    frame_delta = cv2.absdiff(frame1, frame2)

    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        temp = cv2.contourArea(c)
        if temp < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh