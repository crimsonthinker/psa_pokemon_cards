import numpy as np
import cv2
from matplotlib.image import imread
import math
import logging
import os
from tqdm import tqdm
import ray
from ray.actor import ActorHandle
from scipy.ndimage import rotate
from tqdm import tqdm
import json
from typing import Tuple
import glob
import collections
import random
import pandas as pd

from utils.utilities import ensure_dir
from models.unet import UNET
from tensorflow import convert_to_tensor
from utils.utilities import get_logger
from utils.ray_progress_bar import ProgressBar

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
		train_directory : str = None,
		grade_path : str = None,
		origin_img_height : int = None,
		origin_img_width : int = None,
		enable_ray : bool = False):
		self._logger = get_logger("VGG16Preprocessor")
		self.shape = (origin_img_height, origin_img_width, 3) # Default shape
		self.feed_size = (405, 506) # feed shape
		self._enable_ray = enable_ray
		if not self._enable_ray: # since RAY cannot pickle weakref object
			self.model = UNET() #Unet model
			self.model.load_weights(self.pretrained_model_path)
		self.pretrained_model_path = os.path.join('checkpoint/cropper/pretrained/checkpoint')
		self._train_directory = train_directory
		self._preprocessed_dataset_path = 'preprocessed_data'
		ensure_dir(self._preprocessed_dataset_path)

		
		self.failed_images_identifiers = [] # list of unsucessfully preprocessed images

		self._identifiers = [] # list of identifiers (id)

		self._grade_path = grade_path
		self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
		self._grades.index = [str(x) for x in self._grades.index]

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
		origin_image = cv2.resize(image[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
		float_origin_image = origin_image.astype(np.float32)
		## Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			inputs_img[0][:506, :405][:, :, i]	= (float_origin_image[:, :, i] - float_origin_image[:, :, i].mean()) / np.sqrt(float_origin_image[:, :, i].var() + 0.001)
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
		box = self.rearrange_box(box) * self.shape[1] // 405

		box[:, 1] = box[:, 1] + self.shape[0]//4
		
		# Cropped image based on the mask
		PSA_WIDTH = np.linalg.norm(box[1]-box[0])
		PSA_HEIGHT = np.linalg.norm(box[3]-box[0])
		aligned_box = np.float32([[0, 0], [PSA_WIDTH, 0], [PSA_WIDTH, PSA_HEIGHT], [0, PSA_HEIGHT]])
		M = cv2.getPerspectiveTransform(box, aligned_box)
		cropped_img = cv2.warpPerspective(image, M, (math.ceil(PSA_WIDTH), math.ceil(PSA_HEIGHT)))
	
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
		card = None
		card = self.crop_image(image)
		ratio = (card.shape[0]/card.shape[1])
		if ratio < 1.3 or ratio > 1.5:
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

	def preprocess(self, front_image : np.ndarray, back_image : np.ndarray, score_type : str) -> Tuple[np.ndarray, np.ndarray]:
		"""Preprocess image by cropping content out of contour and merge image.
		For each aspect, different preprocessing approach is used on the image


		Args:
			front_image (np.ndarray): front image of the card
			back_image (np.ndarray): back image of the card
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
		Return:
			(np.ndarray, np.ndarray) : A preprocessed image, with residual (if exist)
		"""

		def pad_card(image : np.ndarray, desire_shape : tuple) -> np.ndarray:
			"""Pad card image to appropriate size

			Args:
				image (np.ndarray): card image
				desire_shape (tuple): desired shape. Format as (height, width)

			Returns:
				np.ndarray: padded card image
			"""
			if image.ndim == 2:
				height, width = image.shape
			else:
				height, width, _ = image.shape
			desire_height = desire_shape[0]
			desire_width = desire_shape[1]
			top = int((desire_height - height) / 2)
			bottom = int((desire_height - height) / 2) + ((desire_height - height) % 2)
			left = int((desire_width - width) / 2)
			right = int((desire_width - width) / 2) + ((desire_width - width) % 2)

			return cv2.copyMakeBorder(
				image, 
				top, 
				bottom, 
				left, 
				right, 
				cv2.BORDER_CONSTANT,
				None,
				value = 0
			)

		def extract_card(card : np.ndarray, score_type : str) -> np.ndarray:
			"""Extract card content using VGG16Preprocessor and preprocess it to appropriate format.

			Args:
				card (np.ndarray): card image
				score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.

			Returns:
				np.ndarray: cropped card image
			"""

			# format card
			if card is not None:
				if score_type == 'Corners':
					top_left = card[:200,:200, :]
					top_right = card[:200,-200:,:]
					bottom_left = card[-200:,:200,:]
					bottom_right = card[-200:,-200:,:]
					top = np.concatenate((top_left, top_right), axis = 1)
					bottom = np.concatenate((bottom_left, bottom_right), axis = 1)
					card = np.concatenate((top, bottom), axis = 0)
					edg = cv2.Canny(card, 100, 200)
					edg[100:300,100:300] = 0 # remove centering part
				elif score_type == 'Edges' or score_type == 'Centering':
					left = rotate(card[:,:200,:], 180)
					right = card[:,-200:,:]
					top = rotate(card[:200,:,:],-90)
					bottom = rotate(card[-200:,:,:],90)
					max_height = max(left.shape[0],right.shape[0], top.shape[0], bottom.shape[0])
					max_width = max(left.shape[1],right.shape[1], top.shape[1], bottom.shape[1])
					obj = tuple([pad_card(x, (max_height, max_width)) for x in [left, right, top, bottom]])
					card = np.concatenate(obj, axis = 1)
					edg = cv2.Canny(card, 100, 200)
				elif score_type == 'Surface':
					edg = cv2.Canny(card, 100, 200)
			return card, edg

		
		front_card, front_residual = extract_card(front_image, score_type)
		back_card, back_residual = extract_card(back_image, score_type)

		# If the aspect is 'Centering', only use the back image.
		# Other wise merge the card
		if score_type != 'Centering':
			if front_card is not None and back_card is not None:
				# merge two image
				front_height, front_width, _ = front_card.shape
				back_height, back_width, _ = back_card.shape
				max_height = max(front_height, back_height)
				max_width = max(front_width, back_width)
				front_card = pad_card(front_card, (max_height, max_width))
				back_card = pad_card(back_card, (max_height, max_width))
				
				merge_image = np.concatenate((front_card, back_card), axis = 1) # merge
				if front_residual is not None and back_residual is not None:
					front_residual = pad_card(front_residual, (max_height, max_width))
					back_residual = pad_card(back_residual, (max_height, max_width))
					merge_residual = np.concatenate((front_residual, back_residual), axis = 1) # merge
				else:
					merge_residual = None
				return merge_image, merge_residual
		elif back_card is not None:
			return back_card, back_residual

		return None,None

	def _prepare(self, score_type : str):
		"""Split dataset into train and test dataset

		Args:
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
		"""
		# ensure directory
		root_path = os.path.join(self._preprocessed_dataset_path, score_type)
		ensure_dir(root_path)

		#read images and extract card contents
		self._images = []
		self._identifiers = []
		self.failed_images = []
		file_names = list(glob.glob(os.path.join(self._train_directory, '*')))
		if self._enable_ray:
			@ray.remote
			def _format_images(file_names: np.ndarray, pba : ActorHandle):
				self.model = UNET() #Initialize image after pickling object
				self.model.load_weights(self.pretrained_model_path)
				logging.disable(logging.WARNING) 
				os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
				results = []
				for name in file_names:
					try:
						folders = name.split("/")
						identifier = folders[-1]
						back_image = os.path.join(name, "back.jpg")
						front_image = os.path.join(name, "front.jpg")
						back_image = cv2.cvtColor(np.array(imread(back_image)), cv2.COLOR_BGR2RGB)
						front_image = cv2.cvtColor(np.array(imread(front_image)), cv2.COLOR_BGR2RGB)
						front_image = self.crop(front_image)
						back_image = self.crop(back_image)
						# append the preprocessed image
						preprocessed_image, residual = self.preprocess(front_image, back_image, score_type)
						if preprocessed_image is not None:
							resized_preprocessed_image = cv2.resize(preprocessed_image, (224, 224), cv2.INTER_AREA)
							if residual is not None:
								# concat the residual to the image
								residual = cv2.resize(residual, (224, 224), cv2.INTER_AREA)
								residual = np.expand_dims(residual, -1)
								resized_preprocessed_image = np.concatenate([resized_preprocessed_image,residual], axis = 2)
							results.append((identifier, resized_preprocessed_image))
						else:
							results.append((identifier, None))
					except:
						results.append((identifier, None))
					pba.update.remote(1)
					return results
			ray.init()
			num_chunks = int(len(file_names) / 4)
			pb = ProgressBar(num_chunks)
			actor = pb.actor
			results = [_format_images.remote(f_names, actor) for f_names in np.array_split(file_names,num_chunks)]
			pb.print_until_done()
			results = ray.get(results)
			results = np.concatenate(results)
			ray.shutdown()
		else:
			def _format_images(name : str):
				try:
					folders = name.split("/")
					identifier = folders[-1]
					back_image = os.path.join(name, "back.jpg")
					front_image = os.path.join(name, "front.jpg")
					back_image = cv2.cvtColor(np.array(imread(back_image)), cv2.COLOR_BGR2RGB)
					front_image = cv2.cvtColor(np.array(imread(front_image)), cv2.COLOR_BGR2RGB)
					front_image = self.crop(front_image)
					back_image = self.crop(back_image)
					preprocessed_image, residual = self.preprocess(front_image, back_image, score_type)
					# append the preprocessed image
					if preprocessed_image is not None:
						resized_preprocessed_image = cv2.resize(preprocessed_image, (224, 224), cv2.INTER_AREA)
						if residual is not None:
							residual = cv2.resize(residual, (224, 224), cv2.INTER_AREA)
							residual = np.expand_dims(residual, -1)
							resized_preprocessed_image = np.concatenate([resized_preprocessed_image,residual], axis = 2)
						return (identifier, resized_preprocessed_image)
					else:
						return (identifier, None)
				except:
					return (identifier, None)
			results = [_format_images(name) for name in tqdm(file_names)]
		results = [x for x in results if x is not None]
		self._identifiers = [x[0] for x in results if x[1] is not None]
		self._images = [x[1] for x in results if x[1] is not None]
		self.failed_images_identifiers = [x[0] for x in results if x[1] is None]

	def _oversampling(self, score_type, max_examples_per_score = 500):
		"""Perform oversampling to prevent class imbalance

		Args:
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
			max_examples_per_score (int, optional): Maximum examples per class. Defaults to 500.
		"""

		image_map = {self._identifiers[x] : self._images[x] for x in range(len(self._identifiers))}
		scores_table = self._grades[score_type]
		counter = collections.defaultdict(list)
		highest_num_examples = 0

		# get the highest number of examples per class
		for x in self._identifiers:
			score = scores_table.loc[x]
			counter[score].append(x)
			if highest_num_examples < len(counter[score]):
				highest_num_examples = len(counter[score])
		if max_examples_per_score < highest_num_examples:
			highest_num_examples = max_examples_per_score

		# Perform oversampling with the highest_num_examples
		for score in counter:
			if len(counter[score]) < highest_num_examples:
				# add images in self._images and self._identifiers
				for _ in range(highest_num_examples - len(counter[score])):
					add_id = random.choice(counter[score])
					self._images.append(image_map[add_id])
					self._identifiers.append(add_id)

	def _save(self, score_type):
		"""data to preprocessed folder

		Args:
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
		"""
		root_train_path = os.path.join(self._preprocessed_dataset_path, score_type)
		ensure_dir(root_train_path)
		num_repeat = {}
		self._identifiers = [f'{x}_0' for x in self._identifiers]
		for i, train_image in enumerate(self._images):
			identifier,repetition = self._identifiers[i].split("_")
			if identifier in num_repeat:
				repetition = num_repeat[identifier]
				num_repeat[identifier] += 1
			else:
				num_repeat[identifier] = 1
			np.save(os.path.join(root_train_path,f'{identifier}_{repetition}.npy'), train_image)

	def work(self, score_type : str):
		"""Preprocess data

		Args:
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
		"""
		self._logger.info(f"Perform splitting")
		self._prepare(score_type)

		self._logger.info(f"Oversampling data")
		self._oversampling(score_type)

		self._logger.info("Save dataset to folder")
		self._save(score_type)
