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
        self._grades.index = [str(x) for x in self._grades.index]
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
        # shuffle using list of indices
        shuffle_indices = list(range(len(self._images)))
        shuffle(shuffle_indices)

        self._images = [self._images[i] for i in shuffle_indices]
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
        front_card = None
        front_card = extract_front_contour_for_pop_image(front_image)
        if front_card is None:
            front_card = extract_front_contour_for_dim_image(front_image)

        back_card = extract_front_contour_for_pop_image(back_image)
        if back_card is None:
            back_card = extract_front_contour_for_dim_image(back_image)

        if front_card is not None and back_card is not None:
            # merge two image
            front_height, front_width, _ = front_card.shape
            back_height, back_width, _ = back_card.shape
            max_height = max(front_height, back_height)
            max_width = max(front_width, back_width)
            front_top = int((max_height - front_height) / 2)
            front_bottom = int((max_height - front_height) / 2) + ((max_height - front_height) % 2)
            back_top = int((max_height - back_height) / 2)
            back_bottom = int((max_height - back_height) / 2) + ((max_height - back_height) % 2)

            front_left = int((max_width - front_width) / 2)
            front_right = int((max_width - front_width) / 2) + ((max_width - front_width) % 2)
            back_left = int((max_width - back_width) / 2)
            back_right = int((max_width - back_width) / 2) + ((max_width - back_width) % 2)

            front_card = cv2.copyMakeBorder(
                front_card, 
                front_top, 
                front_bottom, 
                front_left, 
                front_right, 
                cv2.BORDER_REPLICATE
            )

            back_card = cv2.copyMakeBorder(
                back_card,
                back_top,
                back_bottom,
                back_left,
                back_right,
                cv2.BORDER_REPLICATE
            )

            merge_image = np.concatenate((front_card, back_card), axis = 1) # merge
            return merge_image
        else:
            return None

    def _save(self):
        """Save data to preprocessed folder
        """
        root_train_path = os.path.join(self._preprocessed_dataset_path, 'train')
        ensure_dir(root_train_path)
        for i, train_image in enumerate(self._train_data):
            rgb_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(root_train_path,f'{self._identifiers[i]}.jpg'), rgb_image)

        root_val_path = os.path.join(self._preprocessed_dataset_path, 'val')
        ensure_dir(root_val_path)
        for i, val_image in enumerate(self._val_data):
            rgb_image = cv2.cvtColor(val_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(root_val_path,f'{self._identifiers[len(self._train_data) + i]}.jpg'), rgb_image)
                
    def _load(self, score_type):
        """Perform loading the images and split into datasets
        """
        autotune = tf.data.AUTOTUNE

        # get labels from train
        train_path = os.path.join(self._preprocessed_dataset_path, 'train')
        train_identifier_list = sorted([Path(name).stem for name in glob.glob(os.path.join(train_path, "*"))])
        train_score_list = [int(self._grades.loc[x][score_type] * 2) for x in train_identifier_list]

        val_path = os.path.join(self._preprocessed_dataset_path, 'val')
        val_identifier_list = sorted([Path(name).stem for name in glob.glob(os.path.join(val_path, "*"))])
        val_score_list = [int(self._grades.loc[x][score_type] * 2) for x in val_identifier_list]

        def _parse(filename : str, label : int):
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize(image_decoded, (self.img_height,self.img_width))
            image = tf.cast(image_resized, tf.float32)
            return image, label

        def _read_dataset(name_list : list, score_list : list):
            data = tf.data.Dataset.from_tensor_slices((
                name_list,
                score_list
            ))
            return data.map(_parse)
        
        self._train_img_ds = _read_dataset(
            train_identifier_list,
            train_score_list
        )

        self.class_names = list(range(0, 21))
        self._train_img_ds = self._train_img_ds.cache().shuffle(100).prefetch(buffer_size = autotune)
        
        self._validation_img_ds = _read_dataset(
            val_identifier_list,
            val_score_list
        )

        self._validation_img_ds = self._validation_img_ds.cache().prefetch(buffer_size = autotune)

        import pdb ; pdb.set_trace()
        

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
            self._save()

        self._logger.info(f"Load images into train and val data")
        self._load(score_type)


if __name__ == '__main__':
    il = ImageLoader(train_directory = 'data', img_height = 256, img_width = 256)
    il.load('Centering')