import collections
import glob
import os
import tensorflow as tf
import random
import uuid
import shutil
from PIL import Image

from utils.utilities import *

class ImageLoader(object):
    """An image preprocessor class for preprocessing image dataset
    and split them into training and testing set for model training

    Args:
        object: python's object type
    """
    def __init__(self, **kwargs):
        """__init__ function
        """
        self._train_directory = kwargs.get('train_directory', os.path.join('temples-train-hard', 'train'))
        self._skip_reloading = kwargs.get('skip_reloading', False)
        self._default_preprocessed_images = '.preprocessed_train_temples'
        if not self._skip_reloading:
            if os.path.exists(self._default_preprocessed_images):
                shutil.rmtree(self._default_preprocessed_images)
            ensure_dir(self._default_preprocessed_images)
        self._logger = get_logger("ImageLoader")

        self._batch_size = kwargs.get('batch_size', 32)
        self._oversampling_ratio_over_max = kwargs.get('oversampling_ratio_over_max', 0.8)
        self._validation_split = kwargs.get('validation_split', 0.3)
        self.img_width = kwargs.get('img_width')
        self.img_height = kwargs.get('img_height')
        self.class_names = []

    def _split(self):
        """Split dataset to train
        """
        # ensure directory
        ensure_dir(self._default_preprocessed_images)
        #read images
        images = []
        for name in glob.glob(os.path.join(self._train_directory, '*', '*')):
            _ , _ , country,_ = name.split("/")
            im = Image.open(name).resize((self.img_width, self.img_height))
            images.append((im, country))
        random.shuffle(images)
        num_of_images = len(images)
        split_thres = int((1.0 - self._validation_split) * num_of_images)
        self._train_data, self._val_data = images[:split_thres], images[split_thres:]
        # save validation data to folder
        for val_image in self._val_data:
            country = val_image[1]
            val_image = val_image[0]
            root_file_path = os.path.join(self._default_preprocessed_images, 'val', country)
            ensure_dir(root_file_path)
            val_image.save(os.path.join(root_file_path, f'{str(uuid.uuid4())}.png'))

    def _oversample(self):
        # oversample on train data
        train_sort = collections.defaultdict(list)
        for train_image, country in self._train_data:
            train_sort[country].append(train_image)
        max_sample_per_class = len(max(train_sort.values(), key = lambda x : len(x)))

        for country in train_sort:
            for train_image in train_sort[country]:
                root_file_path = os.path.join(self._default_preprocessed_images, 'train', country)
                ensure_dir(root_file_path)
                train_image.save(os.path.join(root_file_path, f'{str(uuid.uuid4())}.png'))
            if len(train_sort[country]) < int(max_sample_per_class * self._oversampling_ratio_over_max):
                num_extra_image = int(max_sample_per_class * self._oversampling_ratio_over_max) - len(train_sort)
                random_images = random.choices(train_sort[country], k = num_extra_image)
                for random_image in random_images:
                    root_file_path = os.path.join(self._default_preprocessed_images, 'train', country)
                    ensure_dir(root_file_path)
                    random_image.save(os.path.join(root_file_path, f'{str(uuid.uuid4())}.png'))

    def _preprocess(self):
        pass
                
    def _load(self):
        """Perform loading the images and split into datasets
        """
        autotune = tf.data.AUTOTUNE

        self._train_img_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self._default_preprocessed_images, 'train'),
            label_mode = 'categorical',
            image_size = (self.img_height, self.img_width),
            batch_size = self._batch_size
        )
        self.class_names = self._train_img_ds.class_names
        self._train_img_ds = self._train_img_ds.cache().shuffle(100).prefetch(buffer_size = autotune)
        
        self._validation_img_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self._default_preprocessed_images, 'val'),
            label_mode = 'categorical',
            image_size = (self.img_height, self.img_width),
            batch_size = self._batch_size
        )
        #FIXME: Inconsistent class names
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

    def load(self):
        """Load images from directory and split into sets
        """
        if not self._skip_reloading:
            self._logger.info(f"Perform splitting")
            self._split()

            self._logger.info(f"Oversampling on train data")
            self._oversample()

            self._logger.info("Preprocess data")
            self._preprocess()

        self._logger.info(f"Load images into train and val data")
        self._load()
