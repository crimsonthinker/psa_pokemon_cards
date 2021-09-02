from utils.preprocessor import VGG16PreProcessor
import os
import cv2
import numpy as np
preprocessor = VGG16PreProcessor()

read_path = "/home/reidite/Dataset/PSA/Part_02/images/cgc"
save_path = "/home/reidite/Dataset/PSA/preprocessed"
image_paths = [x[0] for x in os.walk(read_path)][1:]
for img_path in image_paths:
	front_path = os.path.join(img_path, "front.jpg")
	back_path = os.path.join(img_path, "back.jpg")
	front_img = back_img = None
	if not os.path.exists(front_path) or not os.path.exists(back_path):
		continue
	front_img = cv2.imread(front_path, cv2.IMREAD_COLOR).astype(np.float32)
	back_img = cv2.imread(back_path, cv2.IMREAD_COLOR).astype(np.float32)
	if front_img.any() and back_img.any():
		front_result = preprocessor.crop_image(front_img)
		back_result = preprocessor.crop_image(back_img)
	else:
		print(img_path)
	image_name = os.path.split(img_path)[1]
	print(image_name)
	# cv2.imwrite("{}/{}_front.jpg".format(save_path, image_name), front_result)
	# cv2.imwrite("{}/{}_back.jpg".format(save_path, image_name), back_result)