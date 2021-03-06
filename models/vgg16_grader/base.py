import numpy as np
import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import os
import shutil
import glob
import json
import pandas as pd
from typing import Union
import math
import json
from tqdm import tqdm
from abc import abstractclassmethod

from tensorflow.keras.applications import VGG16
from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset

from utils.utilities import *
from utils.constants import *
from task.loaders import GraderImageLoader

class VGG16GraderBase(object):
    """A baseline grading model class for individual aspect

    Args:
        object: python object
    """
    def __init__(self, 
        grade_name : str,
        max_score : int = 10,
        img_height : int = 224, 
        img_width : int = 224, 
        dim : int = 3,
        learning_rate : float = 0.001,
        epochs : int = 10,
        clean_log : bool = False,
        clean_checkpoints : bool = False):
        """Init function

        Args:
            grade_name (str): score type of the model. Currently, they are 'Surface', 'Centering', 'Corners' and 'Edges
            max_score (int, optional): maximum score of the grading system. Defaults to 10.
            img_height (int, optional): height of the input image. Defaults to 224.
            img_width (int, optional): width of the input image. Defaults to 224.
            dim (int, optional): dimension of the image. Defaults to 3.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            epochs (int, optional): number of training rounds. Defaults to 10.
            clean_log (bool, optional): [description]. Defaults to False.
            clean_checkpoints (bool, optional): [description]. Defaults to False.
        """

        self._logger = get_logger(f"VGG16Grader{grade_name}")
        self._model_name = f'vgg16_grader_{grade_name}'

        # remove log folder
        if os.path.exists('.log') and clean_log:
            shutil.rmtree('.log')
        elif not os.path.exists('.log'):
            ensure_dir('.log')

        # clean checkpoints
        if os.path.exists(os.path.join('checkpoint', self._model_name)) and clean_checkpoints:
            shutil.rmtree(os.path.join('checkpoint', self._model_name))
        elif not os.path.exists(os.path.join('checkpoint', self._model_name)):
            ensure_dir(os.path.join('checkpoint', self._model_name))

        self.grade_name = grade_name
        self.max_score = max_score

        self.img_height = img_height
        self.img_width = img_width
        self.dim = dim
        self.learning_rate = learning_rate

        self._epochs = epochs
        self._history = None

        self._construct()

        now = datetime.utcnow().strftime(SQL_TIMESTAMP)
        self._root_path = os.path.join('checkpoint', self._model.name, now)

        # add earlyStopping
        # use val_loss for monitoring
        # minimize the val_loss
        # print the epoch by setting verbose as 1
        self._cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = self._root_path, verbose = 1, save_best_only = True)
        self._tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./.log/tensorboard', histogram_freq=1)

    @abstractclassmethod
    def _define_meaty_layer(self, inputs):
        """Define main layer for each aspect of the card
        """
        pass

    @abstractclassmethod
    def _define_remain_layer(self,inputs):
        """Define preprocess layer
        """
        pass

    def _construct(self):
        """Construct main model
        """

        self._inputs = tf.keras.Input(shape = (self.img_height, self.img_width, self.dim))

        # Split the image into 2 flows
        if self.dim > 3:
            # Create an extra flow to extract tensor outside of rgb image
            self._image_sliced = tf.keras.layers.Lambda(lambda x : x[:,:,:,:3])(self._inputs)
            self._remains = tf.keras.layers.Lambda(lambda x : x[:,:,:,3:])(self._inputs)
        else:
            self._image_sliced = self._inputs
            self._remains = None

        # data augmentation part
        self._random_flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(self._image_sliced)
        self._random_rotation = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)(self._random_flip)
        self._random_contrast = tf.keras.layers.experimental.preprocessing.RandomContrast(0.3)(self._random_rotation)

        self._base_model = VGG16(weights = 'imagenet', include_top = False)
        # freeze the layer in VGG16. Propagate, flatten and concatenate the output
        for layer in self._base_model.layers:
            layer.trainable = False
        self._base_model_outputs = self._base_model(self._random_contrast)
        self._flatten_input = tf.keras.layers.Flatten()(self._base_model_outputs)
        if self._remains is not None:
            self._flatten_remains = self._define_remain_layer(self._remains)
            self._flatten_input = tf.keras.layers.Concatenate(axis = -1)([self._flatten_input, self._flatten_remains])
        # go through several layers to output the final scores
        self._outputs = self._define_meaty_layer(self._flatten_input)

        self._model = tf.keras.Model(inputs = self._inputs, outputs = self._outputs, name = self._model_name)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = self.learning_rate,
            decay_steps=10000,
            decay_rate=0.01
        )

        # using sparse categorical: maximal true value = index of the maximal predicted value
        self._model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr_schedule),
            loss = 'mae',
            metrics = [
                'mean_absolute_error',
                'mse'
            ]
        )

    def get_summary(self):
        """Get summary of the model
        """
        self._model.summary()

    def set_epoch(self, new_epoch : int):
        """Set epoch of the model

        Args:
            new_epoch (int): number of training rounds
        """
        self._epochs = new_epoch

    def train_and_evaluate(self, dataset : GraderImageLoader):
        """Train and evaluate the model

        Args:
            dataset (GraderImageLoader): dataset used for training
        """
        try:
            self._logger.info(f"""
                Traning VGG16ImageGrader:
                    Image resolution (hxw): {self.img_height}x{self.img_width}
                    Grade name: {self.grade_name}
                    Number of epochs: {self._epochs}
                    Maximum score label: {self.max_score}
                Model summary:""")
            self._model.summary()
            self._history = self._model.fit(
                dataset.get_train_ds(),
                validation_data = dataset.get_validation_ds(),
                epochs = self._epochs,
                callbacks = [
                    self._cp_callback,
                    self._tensorboard_callback
                ]
            )
        except KeyboardInterrupt:
            self._logger.info("Interrupt training session")

    def save_history(self):
        """Save training history of the model
        """
        self._logger.info(f"Saving history in {self._root_path}")
        ensure_dir(self._root_path)

        # save class names as pickle
        with open(os.path.join(self._root_path, 'history.json'), 'w') as f:
            json.dump(self._history.history, f, indent = 4)

    def save_metadata(self):
        """Save metadata of the model in checkpoint folder
        """
        # Update new root path
        self._logger.info(f"Saving metadata in {self._root_path}")
        ensure_dir(self._root_path)
        # save other metadata
        metadata = {
            'grade_name' : self.grade_name,
            'max_score' : self.max_score,
            'img_height' : self.img_height,
            'img_width' : self.img_width,
            'dim' : self.dim,
            'learning_rate' : self.learning_rate,
            'epochs' : self._epochs
        }
        # save class names as pickle
        with open(os.path.join(self._root_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent = 4)


    def load(self, timestamp : str = None):
        """Load the checkpoint of the model, given the timestamp

        Args:
            timestamp (str, optional): timestamp of the checkpoint. Defaults to None.
        """
        max_datetime = None
        if timestamp is None:
            # get the lastest checkpoint provided that the timestamp is None
            max_datetime = None
            for f_name in glob.glob(os.path.join("checkpoint", self._model_name, '*')):
                _,_, d_time = f_name.split('/')
                if max_datetime is None:
                    max_datetime = d_time
                elif datetime.strptime(max_datetime, SQL_TIMESTAMP) < datetime.strptime(d_time, SQL_TIMESTAMP):
                    max_datetime = d_time
        else:
            max_datetime = timestamp

        if max_datetime is None:
            return

        # load the model
        self._root_path = os.path.join('checkpoint', self._model_name, max_datetime)
        self._model = tf.keras.models.load_model(self._root_path)
        #read the metadata
        metadata = json.load(open(os.path.join(self._root_path, 'metadata.json'), 'r'))
        self.grade_name = metadata['grade_name']
        self.max_score = metadata['max_score']
        self.img_height = metadata['img_height']
        self.img_width = metadata['img_width']
        self.dim = metadata['dim']
        self.learning_rate = metadata['learning_rate']
        self._epochs = metadata['epochs']

    def get_checkpoint_path(self) -> str:
        """Return path of the checkpoint of the model

        Returns:
            str: path of checkpoint
        """
        return self._root_path

    def predict(self, x : Union[np.ndarray, str, PrefetchDataset], batch_size = 32):
        """Grade card images based on the aspect of the model

        Args:
            x (Union[np.ndarray, str, PrefetchDataset]): Input is either a numpy array of image(s), a string of file paths, or a PrefetchDataset (for evaluation)
            batch_size (int, optional): batch size for prediction. Defaults to 32.

        Returns:
            result
        """
        if isinstance(x, str):
            # glob files
            predicted_filenames = []
            images = []
            image_predicted = 0
            result = pd.Series()
            result.index.name = 'Identifier'
            result.index = result.index.astype(np.int64)
            result.name = self.grade_name
            for name in tqdm(sorted(glob.glob(os.path.join(x, '*')))):
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        name, target_size=(self.img_height, self.img_width)
                    )
                    images.append(tf.keras.preprocessing.image.img_to_array(img))
                    predicted_filenames.append(os.path.basename(name))
                    if len(images) == batch_size:
                        self._logger.info('Start prediction')
                        ims = np.stack(images)
                        predictions = self.predict(ims)
                        image_predicted += len(images)
                        self._logger.info(f"Predicted {image_predicted} images")
                        for t, filename in enumerate(predicted_filenames):
                            result.loc[os.path.splitext(filename)[0]] = predictions[t] 
                        self._logger.info(result.value_counts())
                        predicted_filenames = []
                        images = []
                except:
                    continue

            if len(images) > 0:
                self._logger.info('Start prediction')
                ims = np.stack(images)
                predictions = self.predict(ims)
                image_predicted += len(images)
                self._logger.info(f"Predicted {image_predicted} images")
                for t, filename in enumerate(predicted_filenames):
                    result.loc[os.path.splitext(filename)[0]] = predictions[t] 

            return result
        elif isinstance(x, np.ndarray):
            if x.ndim == 3:
                # expand dimension of one image to (1, img_height, img_width, 3) -> ndim == 4
                x = np.expand_dims(x, 0)
            if len(x) > batch_size:
                chunks = np.array_split(x, math.ceil(len(x) / batch_size))
                predictions = []
                for chunk in chunks:
                    p = self._model.predict(chunk)
                    predictions.append(p)
            else:
                predictions = [self._model.predict(x)]
            predictions = [np.squeeze(p, axis = -1) * 10.0 for p in predictions]
            predictions = np.hstack(predictions)

            return predictions
        else:
            # not using batch_size from function since PrefetchDataset already has batch_size
            predictions = self._model.predict(x)
            predictions = [np.squeeze(p, axis = -1) * 10.0 for p in predictions]
            return predictions
            



