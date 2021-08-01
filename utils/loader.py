from matplotlib.image import imread, imsave
import glob
import os
import cv2
import tensorflow as tf
import random
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from random import shuffle

from utils.utilities import *
from utils.preprocessor import *

class ImageLoader(object):
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
        validation_split [float] : ratio for validation split
        img_width [float] : image width for the model's input
        img_height [float] : image height for the model's input
        """
        self._train_directory = kwargs.get('train_directory', None)
        self._grade_path = kwargs.get('grade_path', os.path.join('data', 'grades.csv'))
        self._skip_preprocessing = kwargs.get('skip_reloading', False)
        self._preprocessed_dataset_path = '.preprocessed_train'
        if not self._skip_preprocessing:
            if os.path.exists(self._preprocessed_dataset_path):
                shutil.rmtree(self._preprocessed_dataset_path)
            ensure_dir(self._preprocessed_dataset_path)
        self._logger = get_logger("ImageLoader")

        self._batch_size = kwargs.get('batch_size', 32)
        self._oversampling_ratio_over_max = kwargs.get('oversampling_ratio_over_max', 0.8)
        self._validation_split = kwargs.get('validation_split', 0.3)
        self._images = None
        self._images_file_name = None
        self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
        self._identifiers = []


        self.img_width = kwargs.get('img_width')
        self.img_height = kwargs.get('img_height')

    def _split(self):
        """Split dataset into train and test dataset
        """
        # ensure directory
        ensure_dir(self._preprocessed_dataset_path)
        #read images
        self._images = []
        self._identifiers = []
        i = 0

        for name in glob.glob(os.path.join(self._train_directory, '*')):
            # check if name is a folder
            if os.path.isdir(name):
                _, identifier = name.split("/")
                back_image = os.path.join(name, "back.jpg")
                front_image = os.path.join(name, "front.jpg")
                back_image = imread(back_image)
                front_image = imread(front_image)

                preprocessed_image = self._preprocess(back_image, front_image)

                # append the preprocessed image
                if preprocessed_image is not None:
                    self._images.append(preprocessed_image)
                    self._identifiers.append(identifier)
            i = i + 1
            if i == 50:
                break
        max_height = max([x.shape[0] for x in self._images])
        max_width = max([x.shape[1] for x in self._images])

        self._images = [np.pad(a, ((0, max_height - a.shape[0]), (0, max_width - a.shape[1]), (0,0)), 'constant', constant_values=0) for a in self._images]

        # shuffle using list of indices
        shuffle_indices = list(range(len(self._images)))
        shuffle(shuffle_indices)

        # convert list to numpy
        self._images = np.array(self._images)[shuffle_indices]
        self._identifiers = [self._identifiers[i] for i in shuffle_indices]

        num_of_images = len(self._images)
        split_thres = int((1.0 - self._validation_split) * num_of_images)
        self._train_data, self._val_data = self._images[:split_thres], self._images[split_thres:]

    def _preprocess(self, back_image : np.ndarray, front_image : np.ndarray):
        """Preprocess image by cropping content out of contour and merge image

        Args:
            back_image (np.ndarray): [description]
            front_image (np.ndarray): [description]
        Return:
            (np.ndarray) : A preprocessed image
        """
        result = extract_front_contour(front_image)
        return result

    def _save(self, score_type):
        """Save data to preprocessed folder
        """
        root_train_path = os.path.join(self._preprocessed_dataset_path, score_type, 'train')
        ensure_dir(root_train_path)
        for i, train_image in enumerate(self._train_data):
            cv2.imwrite(os.path.join(root_train_path,f'{self._identifiers[i]}.jpg'), train_image)

        root_val_path = os.path.join(self._preprocessed_dataset_path, score_type, 'val')
        ensure_dir(root_val_path)
        for i, val_image in enumerate(self._val_data):
            cv2.imwrite(os.path.join(root_val_path,f'{self._identifiers[len(self._train_data) + i]}.jpg'), val_image)
                
    def _load(self, score_type):
        """Perform loading the images and split into datasets
        """
        autotune = tf.data.AUTOTUNE

        # get labels from train
        train_path = os.path.join(self._preprocessed_dataset_path, 'train')
        train_identifier_list = sorted([Path(name).stem for name in glob.glob(train_path)])

        val_path = os.path.join(self._preprocessed_dataset_path, 'val')
        val_identifier_list = sorted([Path(name).stem for name in glob.glob(val_path)])

        self._train_img_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_path,
            labels = self._grades.loc[train_identifier_list][score_type],
            label_mode = 'categorical',
            image_size = (self.img_height, self.img_width),
            batch_size = self._batch_size
        )
        self.class_names = self._train_img_ds.class_names
        self._train_img_ds = self._train_img_ds.cache().shuffle(100).prefetch(buffer_size = autotune)
        
        self._validation_img_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_path,
            labels = self._grades.loc[val_identifier_list][score_type],
            label_mode = 'categorical',
            image_size = (self.img_height, self.img_width),
            batch_size = self._batch_size
        )

        self._validation_img_ds = self._validation_img_ds.cache().prefetch(buffer_size = autotune)
        

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

    def load(self, score_type):
        """Load images from directory and split into sets
        """

        if not self._skip_preprocessing:
            self._logger.info(f"Perform splitting")
            self._split()

            self._logger.info("Save dataset to folder")
            self._save(score_type)

        self._logger.info(f"Load images into train and val data")
        # self._load(score_type)


if __name__ == '__main__':
    il = ImageLoader(train_directory = 'data')
    il.load('Centering')