import collections
from matplotlib.image import imread
import glob
import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from pathlib import Path
from tqdm import tqdm
import ray
import traceback
from ray.actor import ActorHandle
import random

from utils.ray_progress_bar import ProgressBar
from utils.utilities import *
from utils.preprocessor import VGG16PreProcessor

class GraderImageLoader(object):
	"""An image preprocessor class for preprocessing image dataset
	and split them into training and testing set for model training

	Args:
		object: python's object type
	"""
	def __init__(self, **kwargs):
		"""initialization function for Image Loader. Containing the following parameters
		train_directory [string] : path to the original training folder
		grade_path [string] : path to the grade file. Defaults as data/grades.csv
		skip_preprocessing [bool] : Indicating whether to skip the preprocessing part 
		batch_size [int] : batch size for the dataset. Defaults as 32.
		oversampling_ratio_over_max [float] : A sampling ratio between the number of images in a label and the maximum number of images in one label.
		val_ratio [float] : ratio for validation split
		img_width [float] : image width for the model's input
		img_height [float] : image height for the model's input
		"""
		self._train_directory = kwargs.get('train_directory', None)
		self._grade_path = kwargs.get('grade_path', os.path.join('data', 'grades.csv'))
		self._preprocessed_dataset_path = 'preprocessed_data'
		ensure_dir(self._preprocessed_dataset_path)
		self._logger = get_logger("GraderImageLoader")

		self._batch_size = kwargs.get('batch_size', 32)
		self._val_ratio = kwargs.get('val_ratio', 0.3)
		self._enable_ray = kwargs.get('enable_ray', False)
		self._images = None
		self._images_file_name = None
		self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
		self._grades.index = [str(x) for x in self._grades.index]
		self._paths = []
		self._identifiers = []
		self._identifier_scores = []
		self._train_paths = []
		self._train_identifiers = []
		self._train_identifier_scores = []
		self._val_paths = []
		self._val_identifiers = []
		self._val_identifier_scores = []
		self.failed_images_identifiers = []
		self.max_score = 10

		self.img_width = kwargs.get('img_width')
		self.img_height = kwargs.get('img_height')

		self.preprocessor = VGG16PreProcessor()

	def _split_and_preprocess(self, score_type):
		"""Split dataset into train and test dataset
		"""
		# ensure directory
		root_path = os.path.join(self._preprocessed_dataset_path, score_type)
		ensure_dir(root_path)
		#read images
		self._images = []
		self._identifiers = []
		self.failed_images = []
		file_names = list(glob.glob(os.path.join(self._train_directory, '*')))
		if self._enable_ray:
			@ray.remote
			def _extract_contour(name : str, pba : ActorHandle):
				try:
					folders = name.split("/")
					identifier = folders[-1]
					back_image = os.path.join(name, "back.jpg")
					front_image = os.path.join(name, "front.jpg")
					back_image = np.array(imread(back_image))
					front_image = np.array(imread(front_image))
					preprocessed_image = self._preprocess(back_image, front_image, score_type)
					# append the preprocessed image
					if preprocessed_image is not None:
						resized_preprocessed_image = cv2.resize(preprocessed_image, (self.img_width, self.img_height), cv2.INTER_AREA)
						pba.update.remote(1)
						return (identifier, resized_preprocessed_image)
					else:
						pba.update.remote(1)
						return (identifier, None)
				except:
					pba.update.remote(1)
					print(traceback.format_exc())
					return (identifier, None)
			ray.init()
			pb = ProgressBar(len(file_names))
			actor = pb.actor
			results = [_extract_contour.remote(name, actor) for name in file_names]
			pb.print_until_done()
			results = ray.get(results)
			ray.shutdown()
		else:
			def _extract_contour(name : str):
				try:
					folders = name.split("/")
					identifier = folders[-1]
					back_image = os.path.join(name, "back.jpg")
					front_image = os.path.join(name, "front.jpg")
					back_image = cv2.cvtColor(np.array(imread(back_image)), cv2.COLOR_BGR2RGB)
					front_image = cv2.cvtColor(np.array(imread(front_image)), cv2.COLOR_BGR2RGB)
					preprocessed_image = self._preprocess(back_image, front_image, score_type)
					# append the preprocessed image
					if preprocessed_image is not None:
						resized_preprocessed_image = cv2.resize(preprocessed_image, (self.img_width, self.img_height), cv2.INTER_AREA)
						return (identifier, resized_preprocessed_image)
					else:
						return (identifier, None)
				except:
					print(traceback.format_exc())
					return (identifier, None)
			results = [_extract_contour(name) for name in tqdm(file_names)]
		results = [x for x in results if x is not None]
		self._identifiers = [x[0] for x in results if x[1] is not None]
		self._images = [x[1] for x in results if x[1] is not None]
		self.failed_images_identifiers = [x[0] for x in results if x[1] is None]

	def _preprocess(self, back_image : np.ndarray, front_image : np.ndarray, score_type : str):
		"""Preprocess image by cropping content out of contour and merge image

		Args:
			back_image (np.ndarray): [description]
			front_image (np.ndarray): [description]
		Return:
			(np.ndarray) : A preprocessed image
		"""

		def pad_card(image : np.ndarray, desire_shape : tuple):
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

		def extract_card(image : np.ndarray, score_type : str):
			# TODO: Use U-Net to extract card
			# card = UNetSomething(...)
			card = self.preprocessor.crop_image(image)
			if card is None:
				# front card not exist
				# border got messed up
				card_pop = self.preprocessor.extract_contour_for_pop_image(image)
				card_dim = self.preprocessor.extract_contour_for_dim_image(image)
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

			if card is not None:
				if score_type == 'Corners':
					top_left = card[:200,:200, :]
					top_right = card[:200,-200:,:]
					bottom_left = card[-200:,:200,:]
					bottom_right = card[-200:,-200:,:]
					top = np.concatenate((top_left, top_right), axis = 1)
					bottom = np.concatenate((bottom_left, bottom_right), axis = 1)
					card = np.concatenate((top, bottom), axis = 0)
				elif score_type == 'Edges' or score_type == 'Centering':
					left = rotate(card[:,:200,:], 180)
					right = card[:,-200:,:]
					top = rotate(card[:200,:,:],-90)
					bottom = rotate(card[-200:,:,:],90)
					max_height = max(left.shape[0],right.shape[0], top.shape[0], bottom.shape[0])
					max_width = max(left.shape[1],right.shape[1], top.shape[1], bottom.shape[1])
					# pad top and bottom 
					obj = tuple([pad_card(x, (max_height, max_width)) for x in [left, right, top, bottom]])
					card = np.concatenate(obj, axis = 1)
					edg = np.expand_dims(cv2.Canny(card,100, 200), -1)
					card = np.concatenate([card,edg], axis = 2)
				elif score_type == 'Surface':
					# Do nothing
					pass
			return card

		# make sure that the front and the back is writable
		front_card = extract_card(front_image, score_type)
		back_card = extract_card(back_image, score_type)

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
				return merge_image
		elif back_card is not None:
			return back_card

		return None

	def _oversampling(self, score_type, max_examples_per_score = 500):
		image_map = {self._identifiers[x] : self._images[x] for x in range(len(self._identifiers))}
		scores_table = self._grades[score_type]
		counter = collections.defaultdict(list)
		highest_num_examples = 0
		for x in self._identifiers:
			score = scores_table.loc[x]
			counter[score].append(x)
			if highest_num_examples < len(counter[score]):
				highest_num_examples = len(counter[score])
		if max_examples_per_score < highest_num_examples:
			highest_num_examples = max_examples_per_score
		for score in counter:
			if len(counter[score]) < highest_num_examples:
				# add images in self._images and self._identifiers
				for _ in range(highest_num_examples - len(counter[score])):
					add_id = random.choice(counter[score])
					self._images.append(image_map[add_id])
					self._identifiers.append(add_id)

	def _save(self, score_type):
		"""Save data to preprocessed folder
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
			# np.save(os.path.join(root_train_path,f'{identifier}_{repetition}.npy'), train_image)
			cv2.imwrite(os.path.join(root_train_path,f'{identifier}_{repetition}.jpg'), train_image)
				
	def load(self, score_type):
		"""Perform loading the images and split into datasets
		"""
		autotune = tf.data.AUTOTUNE

		self.train_identifier_list = []
		# save current score type
		self.score_type = score_type

		root_path = os.path.join(self._preprocessed_dataset_path, score_type)
		self._paths = sorted([name for name in glob.glob(os.path.join(root_path, "*"))])
		self._identifiers = [Path(name).stem.split("_")[0] for name in self._paths]
		self._identifier_scores = [self._grades.loc[x][score_type] for x in self._identifiers]

		idx_list = list(range(len(self._paths)))
		random.shuffle(idx_list)

		self._paths = [self._paths[i] for i in idx_list]
		self._identifiers = [self._identifiers[i] for i in idx_list]
		self._identifier_scores = [self._identifier_scores[i] for i in idx_list]

		self._train_paths = self._paths[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._train_identifiers = self._identifiers[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._train_identifier_scores = self._identifier_scores[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._val_paths = self._paths[int(len(self._identifiers) * (1 - self._val_ratio)):]
		self._val_identifiers = self._identifiers[int(len(self._identifiers) * (1 - self._val_ratio)):]
		self._val_identifier_scores = self._identifier_scores[int(len(self._identifiers) * (1 - self._val_ratio)):]

		def _read_dataset(name_list : list, score_list : list):
			# scaling score from 0-10 to 0-1
			images = np.array([np.load(filename) for filename in name_list])
			score_list = [x / self.max_score for x in score_list]
			data = tf.data.Dataset.from_tensor_slices((
				images,
				score_list
			))

			return data


		self._train_img_ds = _read_dataset(
			self._train_paths,
			self._train_identifier_scores
		).batch(self._batch_size).cache().prefetch(buffer_size = autotune)

		self._validation_img_ds = _read_dataset(
			self._val_paths,
			self._val_identifier_scores
		).batch(self._batch_size).cache().prefetch(buffer_size = autotune)
		
	def get_train_ds(self):
		"""Get training image dataset

		Returns:
			tf.data.Dataset: A Tensorflow dataset
		"""
		return self._train_img_ds

	def get_validation_ds(self):
		"""Get validation image dataset

		Returns:
			tf.data.Dataset: A Tensorflow dataset
		"""
		return self._validation_img_ds

	def get_score_type(self):
		return self.score_type
	
	def get_val_identifiers(self):
		return self._val_identifiers
		
	def get_val_scores(self):
		return self._val_identifier_scores
	def preprocess(self, score_type):
		"""Preprocess data

		Args:
			score_type ([type]): [description]
		"""
		self._logger.info(f"Perform splitting")
		self._split_and_preprocess(score_type)

		self._logger.info(f"Oversampling data")
		self._oversampling(score_type)

		self._logger.info("Save dataset to folder")
		self._save(score_type)


class UNETDataLoader(object):
	def __init__(self,
		batch_size,
		original_height,
		original_width,
		dim,
		):
		self.batch_size = batch_size
		self.shape = (original_height, original_width, dim)
		self.dim = (405, 506)
		self.data_path = os.path.join("preprocessed_data", "UNET", "train")
		self.image_paths = []

		self.train_paths = []
		self.train_pivot = 0
		self.num_batch4train = -1

		self.test_paths = []
		self.test_pivot = 0
		self.num_batch4test = -1

		self.mask_path = os.path.join("data", "mask.png")
		self.mask = None

		
	def load(self, mask=True):
		"""Load images paths
		"""
		self.image_paths = [x[0] for x in os.walk(self.data_path)][1:]
		if mask:
			mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * 4
			mask_img[mask_img < 1] = 1.0
			mask_img = np.repeat(mask_img[np.newaxis, :, :], self.batch_size, axis=0)
			self.mask = np.zeros([self.batch_size, 512, 512, 1], np.float32)
			self.mask[:, :506, :405] = np.expand_dims(mask_img, -1)
			self.mask = tf.convert_to_tensor(self.mask)
		pass

	def next_train_batch(self):
		inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
		inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

		for i in range(0, self.batch_size, 2):
			img_path = self.train_paths[self.train_pivot]
			(inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
				 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
			self.train_pivot += 1
		return tf.convert_to_tensor(inputs_img), tf.convert_to_tensor(inputs_gtr)

	def next_test_batch(self):
		inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
		inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

		for i in range(self.batch_size//2):
			img_path = self.test_paths[self.test_pivot]
			(inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
				 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
			self.test_pivot += 1
		return tf.convert_to_tensor(inputs_img), tf.convert_to_tensor(inputs_gtr)
	
	def is_enough_data(self, is_train = True):
		num_of_remained_data = len(self.train_paths) - (self.train_pivot) if is_train else len(self.test_paths) - (self.test_pivot)
		return True if num_of_remained_data >= self.batch_size else False
	
	def split(self, ratio : float):
		"""Split data into train and validation set

		Args:
			ratio (float): ratio of validation set
		"""
		cls_lst = self.image_paths.copy()
		random.shuffle(cls_lst)
		split_idx = int(ratio*len(cls_lst))
		self.train_paths = cls_lst[:split_idx]
		self.num_batch4train = len(self.train_paths) // self.batch_size
		self.test_paths = cls_lst[split_idx:]
		self.num_batch4test = len(self.test_paths) // self.batch_size
	
	def shuffle(self):
		"""Shuffling path lists
		"""
		random.shuffle(self.train_paths)
		random.shuffle(self.test_paths)

	def reset(self):
		self.train_pivot = 0
		self.test_pivot = 0

	def dataparser(self, img_path):
		input_f = cv2.imread(os.path.join(img_path, "front.jpg"), cv2.IMREAD_COLOR).astype(np.float32)
		input_b = cv2.imread(os.path.join(img_path, "back.jpg"), cv2.IMREAD_COLOR).astype(np.float32)
		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			input_f[:, :, i]	=	(input_f[:, :, i] - input_f[:, :, i].mean()) / np.sqrt(input_f[:, :, i].var() + 0.001)
			input_b[:, :, i]	=	(input_b[:, :, i] - input_b[:, :, i].mean()) / np.sqrt(input_b[:, :, i].var() + 0.001)
		gtr_f = cv2.imread(os.path.join(img_path, "gtr_front.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
		gtr_b = cv2.imread(os.path.join(img_path, "gtr_back.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
		return [input_f, input_b], [np.expand_dims(gtr_f, -1), np.expand_dims(gtr_b, -1)]