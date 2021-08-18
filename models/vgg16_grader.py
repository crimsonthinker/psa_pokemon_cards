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

from tensorflow.keras.applications import VGG16

from utils.utilities import *
from utils.constants import *
from utils.loader import *

class VGG16Grader(object):
    def __init__(self, 
        grade_name : str,
        max_score : int = 10,
        img_height : int = 256, 
        img_width : int = 256, 
        dim : int = 3,
        learning_rate : float = 0.001,
        epochs : int = 10,
        clean_log : bool = False,
        clean_checkpoints : bool = False):

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

        # only for training
        self._data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal", 
                input_shape=(img_height, img_width,dim)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.3),
        ], name = 'data_augmentation_layers')

        self._base_model = VGG16(
            weights = 'imagenet',
            include_top = False
        )

        self._layer_only = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, 
                activation = 'relu',
                kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, 
                activation = 'relu',
                kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ], name = 'meaty_layer')

        self._construct()

        now = datetime.utcnow().strftime(SQL_TIMESTAMP)
        self._root_path = os.path.join('checkpoint', self._model.name, now)

        # add earlyStopping
        # use val_loss for monitoring
        # minimize the val_loss
        # print the epoch by setting verbose as 1
        self._cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = self._root_path, verbose = 1, save_best_only = True)
        self._tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./.log/tensorboard', histogram_freq=1)

    def _construct(self):
        # freeze the layer in VGG16
        for layer in self._base_model.layers:
            layer.trainable = False

        self._model = tf.keras.Sequential([
            self._data_augmentation,
            self._base_model,
            self._layer_only
        ], name = self._model_name)

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
        self._model.summary()

    def set_epoch(self, new_epoch : int):
        self._epochs = new_epoch

    def train_and_evaluate(self, dataset : GraderImageLoader):
        try:
            self._logger.info(f"""
                Traning VGG16ImageGrader:
                    Image resolution (hxw): {self.img_height}x{self.img_width}
                    Grade name: {self.grade_name}
                    Number of epochs: {self._epochs}
                    Maximum score label: {self.max_score}
                Model summary:""")
            self._model.summary()
            self._data_augmentation.summary()
            self._layer_only.summary()
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
        self._logger.info(f"Saving history in {self._root_path}")
        ensure_dir(self._root_path)

        # save class names as pickle
        with open(os.path.join(self._root_path, 'history.json'), 'w') as f:
            json.dump(self._history.history, f, indent = 4)

    def save_metadata(self):
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
        """Load the newest checkpoints in the folder
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
        model = tf.keras.models.load_model(self._root_path)
        self._data_augmentation = model.layers[0]
        self._base_model = model.layers[1]
        self._layer_only = model.layers[2]
        #read the metadata
        metadata = json.load(open(os.path.join(self._root_path, 'metadata.json'), 'r'))
        self.grade_name = metadata['grade_name']
        self.max_score = metadata['max_score']
        self.img_height = metadata['img_height']
        self.img_width = metadata['img_width']
        self.dim = metadata['dim']
        self.learning_rate = metadata['learning_rate']
        self._epochs = metadata['epochs']
        # reconstruct the model
        self._construct()

    def predict(self, x : Union[np.ndarray, str], batch_size = 32):
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
                        print(f"Predicted {image_predicted} images")
                        for t, filename in enumerate(predicted_filenames):
                            result.loc[os.path.splitext(filename)[0]] = predictions[t] 
                        print(result.value_counts())
                        predicted_filenames = []
                        images = []
                except:
                    continue

            if len(images) > 0:
                self._logger.info('Start prediction')
                ims = np.stack(images)
                predictions = self.predict(ims)
                image_predicted += len(images)
                print(f"Predicted {image_predicted} images")
                for t, filename in enumerate(predicted_filenames):
                    result.loc[os.path.splitext(filename)[0]] = predictions[t] 

            return result
        else:
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
            predictions = np.hstack([psa_score(x) for x in predictions])

            return predictions



