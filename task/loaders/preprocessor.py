import numpy as np
import cv2
import math
import os
from tqdm import tqdm
import json

from utils.utilities import ensure_dir
from models.unet import UNET
from tensorflow import convert_to_tensor

class UNETPreProcessor(object):
	"""A preprocessor class for UNET

	Args:
		object: python's object
	"""
	def __init__(self,
		height : int,
		width : int,
		dim : int):
		"""Init function

		Args:
			original_height (int): original height
			original_width (int): original width
			dim (int): image dimension
		"""
		self.train_read_path = os.path.join("unet_labeled", "train")
		self.test_read_path  = os.path.join("unet_labeled", "test")
		self.train_save_path = os.path.join("preprocessed_data", "UNET", "train")
		self.test_save_path  = os.path.join("preprocessed_data", "UNET", "test")

		ensure_dir(self.train_save_path)
		ensure_dir(self.test_save_path)

		self.train_image_paths = []
		self.test_image_paths = []

		self.shape = (height, width, dim)
		self.feed_size = (405, 506) # fixed feed size

	def load(self):
		"""Load unpreprocessed file paths
		"""
		self.train_image_paths = [x[0] for x in os.walk(self.train_read_path)][1:]
		self.test_image_paths = [x[0] for x in os.walk(self.test_read_path)][1:]

	def operate(self):
		"""For each file, preprocess the image
		"""
		for img_path in tqdm(self.train_image_paths):
			self.process(img_path, isTrain=True)
		for img_path in tqdm(self.test_image_paths):
			self.process(img_path, isTrain=False)
	
	def process(self, img_path : str, isTrain : bool = True):
		"""Preprocess image and save to destination folder

		Args:
			img_path (str): image
			isTrain (bool, optional): Is the file from train folder?. Defaults to True.
		"""

		def _preprocess(image : np.ndarray, pos : str):
			"""Preprocess module

			Args:
				image (np.ndarray): image
				pos (str): position. Either 'front' or 'back'

			Returns:
				(np.ndarray, np.ndarray): preprocessed image and
			"""
			origin_shape = image.shape
			image = cv2.resize(image, (self.shape[1], self.shape[0]), interpolation = cv2.INTER_AREA)
			seg = np.zeros([image.shape[0], image.shape[1], 1],dtype=np.float32)
			points = np.array([[pt[0],pt[1]] \
				for pt in json.load(open(os.path.join(img_path, f"{pos}.json")))['shapes'][0]['points']])
			for i in range(len(points)):
				points[i][0] = round(points[i][0]*self.shape[1]/origin_shape[1])
				points[i][1] = round(points[i][1]*self.shape[0]/origin_shape[0])
			cv2.fillPoly(seg, pts =[points.astype(np.int0)], color=(255,255,255))
			inputs = cv2.resize(image[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
			gtr = cv2.resize(seg[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
			return inputs, gtr

		img_name = os.path.split(img_path)[1]

		front_path = os.path.join(img_path, "front.jpg")
		back_path = os.path.join(img_path, "back.jpg")

		front_img = cv2.imread(front_path, cv2.IMREAD_COLOR)
		back_img = cv2.imread(back_path, cv2.IMREAD_COLOR)

		input_f, gtr_f = _preprocess(front_img, 'front')
		input_b, gtr_b = _preprocess(back_img, 'back')

		img_folder = os.path.join(self.train_save_path, img_name) if isTrain else \
					os.path.join(self.test_save_path, img_name)
		if not os.path.isdir(img_folder):
			os.mkdir(img_folder)
		cv2.imwrite("{}/front.jpg".format(img_folder), input_f)
		cv2.imwrite("{}/back.jpg".format(img_folder), input_b)
		cv2.imwrite("{}/gtr_front.jpg".format(img_folder), gtr_f)
		cv2.imwrite("{}/gtr_back.jpg".format(img_folder), gtr_b)

class VGG16PreProcessor(object):
	"""A VGG16 preprocessor class

	Args:
		object ([type]): [description]
	"""
	def __init__(self,
		height : int,
		width : int,
		dim : int):
		self.shape = (height, width, dim) # Default shape
		self.feed_size = (405, 506) # feed shape
		self.model = UNET() #Unet model
		self.pretrained_model_path = os.path.join('checkpoint/cropper/pretrained/checkpoint')
		self.model.load_weights(self.pretrained_model_path)

	def crop_image(self, image : np.ndarray) -> np.ndarray:
		"""Crop image to feed to UNet

		Args:
			image (np.ndarray): image

		Returns:
			np.ndarray: cropped image
		"""
		# Resize to a fix shape
		image = cv2.resize(image, (self.shape[1], self.shape[0]), interpolation = cv2.INTER_AREA)
		# Feed image to image to get mask
		inputs_img = np.zeros([1, 512, 512, 3], np.float32)
		origin_image = cv2.resize(image[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA).astype(np.float32)
		## Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			inputs_img[0][:506, :405][:, :, i]	= (origin_image[:, :, i] - origin_image[:, :, i].mean()) / np.sqrt(origin_image[:, :, i].var() + 0.001)
		preds = self.model(convert_to_tensor(inputs_img), training=False).numpy()
		pred_mask = preds[0][:506, :405]
		pred_mask[pred_mask>=0.5] = 1.0
		pred_mask[pred_mask<0.5] = 0.0

		# Get the most appropriate mask
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(16,27))
		pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
												 
		contours, _ = cv2.findContours(image=pred_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
		c = np.concatenate(contours, axis=0)
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = self.rearrange_box(box)

		# Cropped image based on the mask
		PSA_WIDTH = np.linalg.norm(box[1]-box[0])
		PSA_HEIGHT = np.linalg.norm(box[3]-box[0])
		aligned_box = np.float32([[0, 0], [PSA_WIDTH, 0], [PSA_WIDTH, PSA_HEIGHT], [0, PSA_HEIGHT]])
		M = cv2.getPerspectiveTransform(box, aligned_box)
		cropped_img = cv2.warpPerspective(origin_image, M, (math.ceil(PSA_WIDTH), math.ceil(PSA_HEIGHT)))

		return cropped_img

	def rearrange_box(self, box):
		"""Rearrange box
		"""
		pts = np.zeros_like(box)
		cons = 300.0
		for p in box:
			if p[0] <= cons and p[1] <= cons:
				pts[0] = p
			elif p[1] <= cons:
				pts[1] = p
			elif p[0] <= cons:
				pts[3] = p
			else:
				pts[2] = p
		return pts

	def crop_card_for_light_image(self, image : np.ndarray):
		"""Extract cards from images with high lighting

		Args:
			image (np.ndarray): [description]

		Returns:
			np.ndarray: output
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

	def crop_card_for_dark_image(self, image : np.ndarray):
		"""Extract cards from images with dark light

		Args:
			image (np.ndarray): [description]

		Returns:
			np.ndarray: output
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

	def crop(self,image : np.ndarray) -> np.ndarray:
		"""crop card image

		Args:
			image (np.ndarray): [description]

		Returns:
			np.ndarray: [description]
		"""
		# crop card
		card = self.crop_image(image)
		ratio = (card.shape[0]/card.shape[1])
		if ratio < 1.38  and ratio > 1.41:
			card = None
			card_pop = self.crop_card_for_light_image(image)
			card_dim = self.crop_card_for_dark_image(image)
			if card_pop is not None or card_dim is not None:
				if card_dim is None:
					card = card_pop
				elif card_pop is None:
					card = card_dim
				else:
					if card_dim.shape[0] * card_dim.shape[1] < card_pop.shape[0] * card_pop.shape[1]:
						card = card_dim
					else:
						card = card_pop
		return card
