import numpy as np
import cv2
import math

def extract_front_contour_for_pop_image(image : np.ndarray):
    """Extract cards from images

    Args:
        image (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    # to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get edges
    _, otsu_grad = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # get contours
    contours, _ = cv2.findContours(otsu_grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # image size
    height, width = otsu_grad.shape
    image_area = height * width

    # sort contour index
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    contours_sort = [contours[i] for i in index_sort]

    # get area and perimeter
    contour_peri = [cv2.arcLength(contours_sort[i], True) for i in range(len(index_sort))]
    approx = [cv2.approxPolyDP(contours_sort[i], 0.001 * contour_peri[i], True) for i in range(len(index_sort))]
    bounding_box = [cv2.boundingRect(approx[i]) for i in range(len(index_sort))]
    contour_area = [bounding_box[i][2] * bounding_box[i][3]  for i in range(len(index_sort))]
    is_card = list(filter(lambda x : x >= 0, [i if contour_area[i] >= 0.48 * image_area and contour_area[i] <= 0.65 * image_area else -1 for i in range(len(index_sort))]))

    if len(is_card) > 0:
        is_card_index = is_card[-1]
        x,y,w,h = bounding_box[is_card_index]
        return image[int(y) : int(y + h), int(x) : int(x + w)]
    else:
        return None

def extract_front_contour_for_dim_image(image : np.ndarray):
    """[summary]

    Args:
        image (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    # to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # get edges
    blur = cv2.GaussianBlur(gray, (3, 3), -10)
    adaptive_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,3)
    edges = cv2.Canny(adaptive_binary,100,200)
    binarized_grad = 255 - edges

    # denoises again
    open_binarized_grad = cv2.morphologyEx(
        binarized_grad, 
        cv2.MORPH_OPEN, 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    # get contours
    contours, _ = cv2.findContours(open_binarized_grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # image size
    height, width = binarized_grad.shape
    image_area = height * width

    # sort contour index
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    contours_sort = [contours[i] for i in index_sort]

    # get area and perimeter
    contour_area = [cv2.contourArea(contours_sort[i]) for i in range(len(index_sort))]
    contour_peri = [cv2.arcLength(contours_sort[i], True) for i in range(len(index_sort))]
    approx = [cv2.approxPolyDP(contours_sort[i], 0.001 * contour_peri[i], True) for i in range(len(index_sort))]
    bounding_box = [cv2.boundingRect(approx[i]) for i in range(len(index_sort))]
    is_card = list(filter(lambda x : x >= 0, [i if contour_area[i] >= 0.48 * image_area and contour_area[i] <= 0.65 * image_area else -1 for i in range(len(index_sort))]))

    if len(is_card) > 0:
        is_card_index = is_card[-1]
        x,y,w,h = bounding_box[is_card_index]
        return image[int(y) : int(y + h), int(x) : int(x + w)]
    else:
        return None

def psa_score(p : np.ndarray): # list of scores, range from 0 to 10
    # round by PSA scoring system
    lower = np.floor(p)
    decimal = p - lower

    for i in range(len(p)):
        if decimal[i] <= 0.25:
            p[i] = math.floor(p[i])
        elif decimal[i] > 0.25 and decimal[i] <= 0.5:
            p[i] = math.floor(p[i]) + 0.5
        elif decimal[i] > 0.5:
            p[i] = math.ceil(p[i])

    return p