import collections
from logging import root
from pdb import set_trace
from matplotlib.image import imread, imsave
import glob
import os
import cv2
import tensorflow as tf
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

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
        self._skip_preprocessing = kwargs.get('skip_preprocessing', False)
        self._preprocessed_dataset_path = '.preprocessed_train'
        if not self._skip_preprocessing:
            if os.path.exists(self._preprocessed_dataset_path):
                shutil.rmtree(self._preprocessed_dataset_path)
            ensure_dir(self._preprocessed_dataset_path)
        self._logger = get_logger("ImageLoader")

        self._batch_size = kwargs.get('batch_size', 32)
        self._validation_split = kwargs.get('validation_split', 0.3)
        self._images = None
        self._images_file_name = None
        self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
        self._grades.index = [str(x) for x in self._grades.index]
        self._identifiers = []
        self.max_score = 10

        self.img_width = kwargs.get('img_width')
        self.img_height = kwargs.get('img_height')

    def _split_and_preprocess(self, score_type):
        """Split dataset into train and test dataset
        """
        # ensure directory
        root_path = os.path.join(self._preprocessed_dataset_path, score_type)
        ensure_dir(root_path)
        #read images
        self._images = []
        self._identifiers = []

        def _extract_contour(name):
            try:
                _, identifier = name.split("/")
                back_image = os.path.join(name, "back.jpg")
                front_image = os.path.join(name, "front.jpg")
                back_image = imread(back_image)
                front_image = imread(front_image)
                preprocessed_image = self._preprocess(back_image, front_image, score_type)

                # append the preprocessed image
                if preprocessed_image is not None:
                    resized_preprocessed_image = cv2.resize(preprocessed_image, (self.img_width, self.img_height), cv2.INTER_AREA)
                    return (identifier, resized_preprocessed_image)
                else:
                    return (identifier, None)
            except:
                return None
        file_names = list(glob.glob(os.path.join(self._train_directory, '*')))
        for i in tqdm(range(0, len(file_names))):
            name = file_names[i]
            x = _extract_contour(name)
            if x is not None:
                if x[1] is not None:
                    self._identifiers.append(x[0])
                    self._images.append(x[1])


    def _preprocess(self, back_image : np.ndarray, front_image : np.ndarray, score_type : str):
        """Preprocess image by cropping content out of contour and merge image

        Args:
            back_image (np.ndarray): [description]
            front_image (np.ndarray): [description]
        Return:
            (np.ndarray) : A preprocessed image
        """
        front_card = None
        if score_type != 'Centering':
            front_card = extract_front_contour_for_pop_image(front_image)
            if front_card is None:
                front_card = extract_front_contour_for_dim_image(front_image)

        back_card = extract_front_contour_for_pop_image(back_image)
        if back_card is None:
            back_card = extract_front_contour_for_dim_image(back_image)

        if front_card is not None and back_card is not None:
            if score_type != 'Surface':
                # Except for the surface, remove the center
                front_card[150:-150, 150:-150] = 0
                back_card[150:-150,150:-150] = 0
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
        elif back_card is not None and score_type == 'Centering':
            # Remove a part of card in the center
            # focus only at the corners and edges
            back_card[150:-150,150:-150] = 0
            return back_card
        else:
            return None

    def _oversampling(self, score_type):
        image_map = {self._identifiers[x] : self._images[x] for x in range(len(self._identifiers))}
        scores_table = self._grades[score_type]
        counter = collections.defaultdict(list)
        highest_num_examples = 0
        for x in self._identifiers:
            score = scores_table.loc[x]
            counter[score].append(x)
            if highest_num_examples < len(counter[score]):
                highest_num_examples = len(counter[score])
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
            rgb_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
            identifier,repetition = self._identifiers[i].split("_")
            if identifier in num_repeat:
                repetition = num_repeat[identifier]
                num_repeat[identifier] += 1
            else:
                num_repeat[identifier] = 1
            cv2.imwrite(os.path.join(root_train_path,f'{identifier}_{repetition}.jpg'), rgb_image)
                
    def _load(self, score_type):
        """Perform loading the images and split into datasets
        """
        autotune = tf.data.AUTOTUNE

        # reverse the score during learning since we are using MAPE -> by reversing, small score (which is large score but reversed)
        # will generate larger loss for the model
        root_path = os.path.join(self._preprocessed_dataset_path, score_type)
        train_files = sorted([name for name in glob.glob(os.path.join(root_path, "*"))])
        train_identifier_list = [Path(name).stem.split("_")[0] for name in train_files]
        train_score_list = [self._grades.loc[x][score_type] / self.max_score for x in train_identifier_list]

        def _parse(filename : str, label : float):
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize(image_decoded, (self.img_height,self.img_width))
            image = tf.cast(image_resized, tf.float32)
            return image, tf.convert_to_tensor(label)

        def _read_dataset(name_list : list, score_list : list):
            data = tf.data.Dataset.from_tensor_slices((
                name_list,
                score_list
            ))
            return data.map(_parse)

        self._train_img_ds = _read_dataset(
            train_files,
            train_score_list
        ).shuffle(100)

        train_size = int(len(self._train_img_ds) * (1 - self._validation_split))
        val_size = len(self._train_img_ds) - train_size
        train_img_ds = self._train_img_ds.take(train_size)    
        self._validation_img_ds = self._train_img_ds.skip(train_size).take(val_size)
        self._train_img_ds = train_img_ds

        self._train_img_ds = self._train_img_ds.batch(self._batch_size).cache().prefetch(buffer_size = autotune)
        self._validation_img_ds = self._validation_img_ds.batch(self._batch_size).cache().prefetch(buffer_size = autotune)
        

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
            self._split_and_preprocess(score_type)

            self._logger.info(f"Oversampling data")
            self._oversampling(score_type)

            self._logger.info("Save dataset to folder")
            self._save(score_type)

        self._logger.info(f"Load images into train and val data")
        self._load(score_type)