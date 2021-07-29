# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MiniBatchKMeans
import imutils


# %%
# root_path = os.path.join('data', 'cgc', '3789366028', 'front.jpg')
DATA_FOLDER = "/home/reidite/Dataset/PSA"
SELECTED_PART = "Part_01"
IMG_NAME_LST = ['3789366028', '3727258020',
				'3727258028', '3727258038',
				'3727258070']
SELECTED_IDX = 2
front_img_name = os.path.join(DATA_FOLDER, SELECTED_PART, 'images/cgc', IMG_NAME_LST[SELECTED_IDX], 'front.jpg')
back_img_name = os.path.join(DATA_FOLDER, SELECTED_PART, 'images/cgc', IMG_NAME_LST[SELECTED_IDX], 'back.jpg')


# %%
image = cv2.imread(front_img_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# %%
# threshold it
blur = cv2.GaussianBlur(gray, (3, 3), 0)
adaptive_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,3)
grad_x = cv2.Sobel(adaptive_binary, cv2.CV_16S, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(adaptive_binary, cv2.CV_16S, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)
ret2, binarized_grad = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(cv2.cvtColor(binarized_grad, cv2.COLOR_BGR2RGB))
plt.imsave("./outputs/test.png", binarized_grad)

scangram_row = np.sum(binarized_grad, axis=1)
scangram_column = np.sum(binarized_grad, axis=0)
THRESHOLD_VALUE = 120
ALLOWED_MARGIN = 200
height, width, _ = image.shape
for i in range(ALLOWED_MARGIN):
	if scangram_row[i] > THRESHOLD_VALUE:
		ROI_Row1 = i
		break
for i in range(height - ALLOWED_MARGIN, height)[::-1]:
	if scangram_row[i] > THRESHOLD_VALUE:
		ROI_Row2 = i
		break

for i in range(ALLOWED_MARGIN):
	if scangram_column[i] > THRESHOLD_VALUE:
		ROI_Column1 = i
		break
for i in range(width - ALLOWED_MARGIN, width)[::-1]:
	if scangram_column[i] > THRESHOLD_VALUE:
		ROI_Column2 = i
		break

# %%
binarized_grad = \
			binarized_grad[ROI_Row1:ROI_Row2,ROI_Column1:ROI_Column2]
image = image[ROI_Row1:ROI_Row2,ROI_Column1:ROI_Column2]
plt.imsave("./outputs/test.png", binarized_grad)

# %%

height, width, _ = image.shape
margin = 10
ROI_Row1 = int(height*5/18 - margin)
ROI_Row2 = int(height*17/18 + margin)
ROI_Column1 = int(width*1/10 - margin)
ROI_Column2 = int(width*9/10 + margin)
cropped_binarized_grad = \
			binarized_grad[ROI_Row1:ROI_Row2,ROI_Column1:ROI_Column2]
cropped_image = \
			image[ROI_Row1:ROI_Row2,ROI_Column1:ROI_Column2]
height, width, _ = cropped_image.shape

def normalize(img, isGray=False):
	if isGray:
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
cropped_image = normalize(cropped_image)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.imsave("./outputs/test.png", cropped_image)
# cropped_gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# cropped_gray_image = cv2.bilateralFilter(cropped_gray_image, 9, 150, 75, cv2.BORDER_REPLICATE)
# edged = cv2.Canny(cropped_gray_image, 0, 70)
lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
gray_lab_image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2GRAY)
# gray_lab_image = normalize(gray_lab_image, True)
plt.imsave("./outputs/test.png", gray_lab_image)
edged = cv2.Canny(gray_lab_image, 0, 45)
plt.imsave("./outputs/test.png", edged)

adaptive_binary = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,3)
# ret, adaptive_binary = cv2.threshold(cropped_gray_image,133,255,cv2.THRESH_BINARY)
plt.imsave("./outputs/test.png", adaptive_binary)
# %%
contours, hierarchy = cv2.findContours(image=edged, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
result = cropped_image.copy()
c = np.concatenate(contours, axis=0)
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
max_ = np.amax(box, axis=0)
min_ = np.amin(box, axis=0)
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

PSA_WIDTH = np.linalg.norm(pts1[1]-pts1[0])
PSA_HEIGHT = np.linalg.norm(pts1[3]-pts1[0])
pts2 = np.float32([[0, 0], [PSA_WIDTH, 0], [PSA_WIDTH, PSA_HEIGHT], [0, PSA_HEIGHT]])
M = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(cropped_image, M, (PSA_WIDTH, PSA_HEIGHT))
plt.imsave("./outputs/test.png", result)



# %%



