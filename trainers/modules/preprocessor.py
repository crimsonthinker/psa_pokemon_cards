import numpy as np
import cv2

def normalize(img : np.ndarray, is_gray = False):
	if is_gray:
		max_ = np.amax(img)
		min_ = np.amin(img)
		img[:,:] = ((img[:,:].astype(np.float32) - min_) / (max_ - min_) * 255.0).astype(np.uint8)
	else:
		for i in range(3):
			sample = img[:,:,i]
			max_ = np.amax(sample)
			min_ = np.amin(sample)
			img[:,:,i] = ((img[:,:,i].astype(np.float32) - min_) / (max_ - min_) * 255.0).astype(np.uint8)
	return img

def extract_front_content(front_image : np.ndarray):
    """Extract content of front image

    Args:
        front_image (np.ndarray): An (img_height, img_width, dim) image as numpy array
    """
    gray = cv2.cvtColor(front_image, cv2.COLOR_BGR2GRAY)

    # blur the image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # threshold using adaptive method
    adaptive_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,3)
    # detect edges using Sobel
    grad_x = cv2.Sobel(adaptive_binary, cv2.CV_16S, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(adaptive_binary, cv2.CV_16S, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)
    # threshold again using Otsu
    _, binarized_grad = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scangram_row = np.sum(binarized_grad, axis=1)
    scangram_column = np.sum(binarized_grad, axis=0)

    THRESHOLD_VALUE = 120
    ALLOWED_MARGIN = 200

    height, width, _ = front_image.shape

    allowed_scangram_row = scangram_row[:ALLOWED_MARGIN]
    where = np.where(allowed_scangram_row > THRESHOLD_VALUE)
    roi_row_1 = where[0][0] if len(where) > 0 and len(where[0]) > 0 else None    
    allowed_scangram_row = scangram_row[:-ALLOWED_MARGIN]
    where = np.where(allowed_scangram_row > THRESHOLD_VALUE)
    roi_row_2 = where[0][-1] if len(where) > 0 and len(where[0]) > 0 else None    

    allowed_scangram_column = scangram_column[:ALLOWED_MARGIN]
    where = np.where(allowed_scangram_column > THRESHOLD_VALUE)
    roi_column_1 = where[0][0] if len(where) > 0 and len(where[0]) > 0 else None    
    allowed_scangram_column = scangram_column[:-ALLOWED_MARGIN]
    where = np.where(allowed_scangram_column > THRESHOLD_VALUE)
    roi_column_2 = where[0][-1] if len(where) > 0 and len(where[0]) > 0 else None    

    binarized_grad = binarized_grad[roi_row_1:roi_row_2, roi_column_1:roi_column_2]
    cropped_image = front_image[roi_row_1:roi_row_2, roi_column_1:roi_column_2,:]


    MARGIN = 10
    height, width, _ = cropped_image.shape
    roi_row_1 = int(height * 5 / 18 - MARGIN)
    roi_row_2 = int(height * 17 / 18 + MARGIN) 
    roi_column_1 = int(width * 1 / 10 - MARGIN)
    roi_column_2 = int(width * 9 / 10 + MARGIN)

    cropped_image = cropped_image[roi_row_1:roi_row_2, roi_column_1:roi_column_2, :]
    height, width, _ = cropped_image.shape
    cropped_image = normalize(cropped_image)

    lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    gray_lab_image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray_lab_image, 0, 45)
    adaptive_binary = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,3)

    contours, _ = cv2.findContours(image = edged, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    result = cropped_image.copy()
    c = np.concatenate(contours, axis=0)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    pts1 = np.zeros_like(box)
    for p in box:
        if p[0] <= 100.0 and p[1] <= 100.0:
            pts1[0] = p
        elif p[1] <= 100.0:
            pts1[1] = p
        elif p[0] <= 100.0:
            pts1[3] = p
        else:
            pts1[2] = p

    PSA_WIDTH = int(np.linalg.norm(pts1[1] - pts1[0]))
    PSA_HEIGHT = int(np.linalg.norm(pts1[3] - pts1[0]))
    pts2 = np.float32([[0, 0], [PSA_WIDTH, 0], [PSA_WIDTH, PSA_HEIGHT], [0, PSA_HEIGHT]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(cropped_image, M, (PSA_WIDTH, PSA_HEIGHT))

    return result