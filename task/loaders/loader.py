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
from typing import Tuple

from utils.ray_progress_bar import ProgressBar
from utils.utilities import *
from task.loaders.preprocessor import VGG16PreProcessor

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
        self._train_directory = kwargs.get('train_directory', None)

        self._grade_path = kwargs.get('grade_path', os.path.join('data', 'grades.csv'))
        self._grades = pd.read_csv(self._grade_path, index_col = 'Identifier')
        self._grades.index = [str(x) for x in self._grades.index]

        self._preprocessed_dataset_path = 'preprocessed_data'
        ensure_dir(self._preprocessed_dataset_path)

        # Fixed ratio obtained from Pokemon PSA card
        self.img_width = kwargs.get('img_width', 128)
        self.img_height = kwargs.get('img_height', 215)

        self.max_score = 10

        self._logger = get_logger("GraderImageLoader")

        self._batch_size = kwargs.get('batch_size', 8)
        self._val_ratio = kwargs.get('val_ratio', 0.3)
        self._enable_ray = kwargs.get('enable_ray', False)
        self._images = None
        self._images_file_name = None

        self._paths = [] # list of file paths
        self._identifiers = [] # list of identifiers (id)
        self._identifier_scores = [] # list of identifier's score (of one aspect)
        self._train_paths = [] # list of train file paths
        self._train_identifiers = [] # list of train identifiers
        self._train_identifier_scores = [] # list of train identifier's score
        self._val_paths = [] # list of validation file path
        self._val_identifiers = [] # list of validation identifiers
        self._val_identifier_scores = [] # list of validation identifier's score
        self.failed_images_identifiers = [] # list of unsucessfully preprocessed images

    def _preprocess(self, score_type : str):
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
                results = []
                cropper = VGG16PreProcessor()
                for name in file_names:
                    try:
                        folders = name.split("/")
                        identifier = folders[-1]
                        back_image = os.path.join(name, "back.jpg")
                        front_image = os.path.join(name, "front.jpg")
                        back_image = np.array(imread(back_image))
                        front_image = np.array(imread(front_image))
                        front_image = cropper.crop(front_image)
                        back_image = cropper.crop(back_image)
                        # append the preprocessed image
                        preprocessed_image, residual = self.__preprocess(front_image, back_image, score_type)
                        if preprocessed_image is not None:
                            resized_preprocessed_image = cv2.resize(preprocessed_image, (self.img_width, self.img_height), cv2.INTER_AREA)
                            if residual is not None:
                                # concat the residual to the image
                                residual = cv2.resize(residual, (self.img_width, self.img_height), cv2.INTER_AREA)
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
            cropper = VGG16PreProcessor()
            def _format_images(name : str):
                try:
                    folders = name.split("/")
                    identifier = folders[-1]
                    back_image = os.path.join(name, "back.jpg")
                    front_image = os.path.join(name, "front.jpg")
                    back_image = cv2.cvtColor(np.array(imread(back_image)), cv2.COLOR_BGR2RGB)
                    front_image = cv2.cvtColor(np.array(imread(front_image)), cv2.COLOR_BGR2RGB)
                    front_image = cropper.crop(front_image)
                    back_image = cropper.crop(back_image)
                    preprocessed_image, residual = self.__preprocess(back_image, front_image, score_type)
                    # append the preprocessed image
                    if preprocessed_image is not None:
                        resized_preprocessed_image = cv2.resize(preprocessed_image, (self.img_width, self.img_height), cv2.INTER_AREA)
                        if residual is not None:
                            residual = cv2.resize(residual, (self.img_width, self.img_height), cv2.INTER_AREA)
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

    def __preprocess(self, front_image : np.ndarray, back_image : np.ndarray, score_type : str) -> Tuple[np.ndarray, np.ndarray]:
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
                    edg = cv2.Canny(card, 50, 150)
                    edg[100:300,100:300] = 0 # remove centering part
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
                    edg = cv2.Canny(card, 50, 150)
                elif score_type == 'Surface':
                    edg = cv2.Canny(card, 50, 150)
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
                front_residual = pad_card(front_residual, (max_height, max_width))
                back_card = pad_card(back_card, (max_height, max_width))
                back_residual = pad_card(back_residual, (max_height, max_width))
                merge_image = np.concatenate((front_card, back_card), axis = 1) # merge
                merge_residual = np.concatenate((front_residual, back_residual), axis = 1) # merge
                return merge_image, merge_residual
        elif back_card is not None:
            return back_card, back_residual

        return None

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

    def preprocess(self, score_type : str):
        """Preprocess data

        Args:
            score_type (string): score aspect for image loader. It is either 'Centering', 'Corners', 'Edges', or 'Surface'.
        """
        self._logger.info(f"Perform splitting")
        self._preprocess(score_type)

        self._logger.info(f"Oversampling data")
        self._oversampling(score_type)

        self._logger.info("Save dataset to folder")
        self._save(score_type)


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
        self.dim = (405, 506) # preprocessed dimension
        self.data_path = os.path.join("preprocessed_data", "UNET")

        self.image_paths = []

        self.train_paths = []
        self.train_pivot = 0
        self.num_batch4train = -1

        self.test_paths = []
        self.test_pivot = 0
        self.num_batch4test = -1

        
    def load(self):
        """Load images paths
        """
        self.image_paths = [x[0] for x in os.walk(self.data_path)][1:]

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
        input_f = cv2.imread(os.path.join(img_path, "front.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        input_b = cv2.imread(os.path.join(img_path, "back.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        gtr_f = cv2.imread(os.path.join(img_path, "gtr_front.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gtr_b = cv2.imread(os.path.join(img_path, "gtr_back.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return [input_f, input_b], [np.expand_dims(gtr_f, -1), np.expand_dims(gtr_b, -1)]