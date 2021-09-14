import glob
import os

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

import random
from PIL import ImageEnhance, Image
from utils.utilities import ensure_dir, get_logger



class GraderImageLoader(object):
	"""An image preprocessor class for preprocessing image dataset
	and split them into training and testing set for grader model

	Args:
		object: python's object type
	"""
	def __init__(self, **kwargs):
		"""initialization function for Image Loader. Containing the following parameters
		train_directory [string] : path to the original training folder
		grade_path [string] : path to the grade file. Defaults as data/grades.csv
		skip_preprocessing [bool] : Indicating whether to skip the preprocessing part 
		batch_size [int] : batch size for the dataset. Defaults as 8.
		val_ratio [float] : ratio for validation split
		enable_ray [bool] : Flag enabling multiprocessing approach for preprocessing module using Ray
		"""
		self._train_directory = kwargs.get('train_directory', '')
		self._grade_path = kwargs.get('grade_path', os.path.join(self._train_directory, 'grades.csv'))
		self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
		self._grades.index = [str(x) for x in self._grades.index]

		self._preprocessed_dataset_path = 'preprocessed_data'
		self.max_score = 10

		self._logger = get_logger("GraderImageLoader")

		self._batch_size = kwargs.get('batch_size', 8)
		self._val_ratio = kwargs.get('val_ratio', 0.3)

		self._images = None
		self._images_file_name = None

		self._paths = [] # list of file paths
		self._identifier_scores = [] # list of identifier's score (of one aspect)
		self._train_paths = [] # list of train file paths
		self._train_identifiers = [] # list of train identifiers
		self._train_identifier_scores = [] # list of train identifier's score
		self._val_paths = [] # list of validation file path
		self._val_identifiers = [] # list of validation identifiers
		self._val_identifier_scores = [] # list of validation identifier's score
			
	def load(self, score_type : str):
		"""Load data from preprocessed_data

		Args:
			score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
		"""

		self.train_identifier_list = []
		self.score_type = score_type

		# prepare path and scores
		root_path = os.path.join(self._preprocessed_dataset_path, score_type)
		self._paths = sorted([name for name in glob.glob(os.path.join(root_path, "*"))])
		self._identifiers = [Path(name).stem.split("_")[0] for name in self._paths]
		self._identifier_scores = [self._grades.loc[x][score_type] for x in self._identifiers]

		# get shuffle indices and shuffle the lists
		idx_list = list(range(len(self._paths)))
		random.shuffle(idx_list)
		self._paths = [self._paths[i] for i in idx_list]
		self._identifiers = [self._identifiers[i] for i in idx_list]
		self._identifier_scores = [self._identifier_scores[i] for i in idx_list]

		# Split to train and validaation
		self._train_paths = self._paths[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._train_identifiers = self._identifiers[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._train_identifier_scores = self._identifier_scores[:int(len(self._identifiers) * (1 - self._val_ratio))]
		self._val_paths = self._paths[int(len(self._identifiers) * (1 - self._val_ratio)):]
		self._val_identifiers = self._identifiers[int(len(self._identifiers) * (1 - self._val_ratio)):]
		self._val_identifier_scores = self._identifier_scores[int(len(self._identifiers) * (1 - self._val_ratio)):]

		# Read dataset
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
		).batch(self._batch_size)
		self._validation_img_ds = _read_dataset(
			self._val_paths,
			self._val_identifier_scores
		).batch(self._batch_size)
		
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
		"""Get score type

		Returns:
			str: name of score aspect
		"""
		return self.score_type
	
	def get_val_identifiers(self):
		"""Get list of validation identifiers

		Returns:
			list[str]: list of validation identifiers
		"""
		return self._val_identifiers
		
	def get_val_scores(self):
		"""Get list of validation scores

		Returns:
			list[str]: list of validation scores
		"""
		return self._val_identifier_scores


class UNETDataLoader(object):
	"""A Data loader class for U-Net

	Args:
		object: python's object type
	"""
	def __init__(self,
		batch_size : int,
		original_height : int,
		original_width : int,
		dim : int,
		):
		"""Init function

		Args:
			batch_size (int): batch_size for data loader
			original_height (int): original height
			original_width (int): original width
			dim (int): image dimension
		"""
		self.batch_size = batch_size
		self.shape = (original_height, original_width, dim)
		self.dim = (405, 506)
		self.train_data_path = os.path.join("preprocessed_data", "UNET", "train")
		self.test_data_path = os.path.join("preprocessed_data", "UNET", "test")
		self.image_paths = []

		self.train_paths = []
		self.train_pivot = 0
		self.num_batch4train = -1

		self.test_paths = []
		self.test_pivot = 0
		self.num_batch4test = -1

		self.mask_path = os.path.join("task/loaders/mask.png")
		self.mask = None

		
	def load(self, mask=True):
		"""Load images paths
		"""
		self.train_paths = [x[0] for x in os.walk(self.train_data_path)][1:]
		self.test_paths = [x[0] for x in os.walk(self.test_data_path)][1:]
		self.image_paths = self.train_paths + self.test_paths
		if mask:
			mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * 4
			mask_img[mask_img < 1] = 1.0
			mask_img = np.repeat(mask_img[np.newaxis, :, :], self.batch_size, axis=0)
			self.mask = np.zeros([self.batch_size, 512, 512, 1], np.float32)
			self.mask[:, :506, :405] = np.expand_dims(mask_img, -1)
			self.mask = tf.convert_to_tensor(self.mask)
		pass

	def next_train_batch(self):
		"""Get next train batch

		Returns:
			(tf.Tensor, tf.Tensor): Tensors for inputs and outputs of U-Net
		"""
		inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
		inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

		for i in range(0, self.batch_size, 2):
			img_path = self.train_paths[self.train_pivot]
			(inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
				 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
			self.train_pivot += 1
		return tf.convert_to_tensor(inputs_img), tf.convert_to_tensor(inputs_gtr)

	def next_test_batch(self):
		"""Get next test batch

		Returns:
			(tf.Tensor, tf.Tensor): Tensors for inputs and outputs of U-Net
		"""
		inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
		inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

		for i in range(self.batch_size//2):
			img_path = self.test_paths[self.test_pivot]
			(inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
				 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
			self.test_pivot += 1
		return tf.convert_to_tensor(inputs_img), tf.convert_to_tensor(inputs_gtr)
	
	def is_enough_data(self, is_train = True) -> bool:
		"""Check if there's data for the module

		Args:
			is_train (bool, optional): check from train data. Defaults to True.

		Returns:
			bool
		"""
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
		"""Reset pivot
		"""
		self.train_pivot = 0
		self.test_pivot = 0

	def dataparser(self, img_path):
		"""parse image file to correct format

		Args:
			img_path (str): image

		Returns:
			([np.ndarray,np,ndarray], [np.ndarray,np,ndarray])]:
				([Front color input, back color input], [front grayscale input, back grayscale input])
		"""
		input_f = cv2.imread(os.path.join(img_path, "front.jpg"), cv2.IMREAD_COLOR).astype(np.float32)
		input_b = cv2.imread(os.path.join(img_path, "back.jpg"), cv2.IMREAD_COLOR).astype(np.float32)

		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			input_f[:, :, i]	=	(input_f[:, :, i] - input_f[:, :, i].mean()) / np.sqrt(input_f[:, :, i].var() + 0.001)
			input_b[:, :, i]	=	(input_b[:, :, i] - input_b[:, :, i].mean()) / np.sqrt(input_b[:, :, i].var() + 0.001)
		gtr_f = cv2.imread(os.path.join(img_path, "gtr_front.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
		gtr_b = cv2.imread(os.path.join(img_path, "gtr_back.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
		return [input_f, input_b], [np.expand_dims(gtr_f, -1), np.expand_dims(gtr_b, -1)]

	def randomColor(self, image):
		PIL_image = Image.fromarray((image).astype(np.uint8))
		random_factor = np.random.randint(0, 11) / 10.
		color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)
		random_factor = np.random.randint(10, 21) / 10.
		brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
		random_factor = np.random.randint(10, 21) / 10.
		contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
		random_factor = np.random.randint(0, 11) / 10.
		out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
		out = out
		return out
