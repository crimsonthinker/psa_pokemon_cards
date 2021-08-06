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
        if os.path.exists(os.path.join('.checkpoints', self._model_name)) and clean_checkpoints:
            shutil.rmtree(os.path.join('.checkpoints', self._model_name))
        elif not os.path.exists(os.path.join('.checkpoints', self._model_name)):
            ensure_dir(os.path.join('.checkpoints', self._model_name))

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

        # freeze the layer in VGG16
        for layer in self._base_model.layers:
            layer.trainable = False

        self._layer_only = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, 
                activation = 'relu',
                kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ], name = 'meaty_layer')

        self._construct()

        now = datetime.utcnow().strftime(SQL_TIMESTAMP)
        self._root_path = os.path.join('.checkpoints', self._model.name, now)

        # add earlyStopping
        # use val_loss for monitoring
        # minimize the val_loss
        # print the epoch by setting verbose as 1
        self._cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = self._root_path, verbose = 1, save_best_only = True)
        self._tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./.log/tensorboard', histogram_freq=1)

    def _construct(self):
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
            loss = 'mse',
            metrics = [
                'mean_absolute_error'
            ]
        )

    def get_summary(self):
        self._model.summary()

    def set_epoch(self, new_epoch : int):
        self._epochs = new_epoch

    def train_and_evaluate(self, dataset : ImageLoader):
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

    def save_visualization_result(self):
        if self._history is not None:
            self._logger.info("Saving visualization result")
            epochs_range = list(range(self._epochs))

            root_path = os.path.join('.log', datetime.utcnow().strftime(SQL_TIMESTAMP))
            ensure_dir(root_path)

            fig = make_subplots(
                rows = 1,
                cols = 2
            )

            fig.add_trace(
                go.Scatter(
                    name = 'Training Accuracy',
                    x = epochs_range, 
                    y = self._history.history['accuracy']),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Validation Accuracy',
                    x = epochs_range, 
                    y = self._history.history['val_accuracy']),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Training Loss',
                    x = epochs_range, 
                    y = self._history.history['loss']),
                row = 1, col = 2
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Validation Loss',
                    x = epochs_range, 
                    y = self._history.history['val_loss']),
                row = 1, col = 2
            )
            fig.update_layout(title = 'Accuracy and Loss')
            fig.write_html(os.path.join(root_path, 'accuracy_and_loss.html'))

            # calculate average precision and recall of model
            total_demoninators = [x + y for x, y in zip(self._history.history['true_positives'], self._history.history['false_positives'])]
            precisions = [denominator / total_demoninator \
                if total_demoninator > 0.0 else 0.0 for denominator, total_demoninator \
                    in zip(self._history.history['true_positives'], total_demoninators)]
            total_demoninators = [x + y for x, y in zip(self._history.history['true_positives'], self._history.history['false_negatives'])]
            recalls = [denominator / total_demoninator \
                if total_demoninator > 0.0 else 0.0 for denominator, total_demoninator \
                    in zip(self._history.history['true_positives'], total_demoninators)]
            total_demoninators = [x + y for x, y in zip(self._history.history['val_true_positives'], self._history.history['val_false_positives'])]
            val_precisions = [denominator / total_demoninator \
                if total_demoninator > 0.0 else 0.0 for denominator, total_demoninator \
                    in zip(self._history.history['val_true_positives'], total_demoninators)]
            total_demoninators = [x + y for x, y in zip(self._history.history['val_true_positives'], self._history.history['val_false_negatives'])]
            val_recalls = [denominator / total_demoninator \
                if total_demoninator > 0.0 else 0.0 for denominator, total_demoninator \
                    in zip(self._history.history['val_true_positives'], total_demoninators)]

            fig = make_subplots(
                rows = 1,
                cols = 2
            )

            fig.add_trace(
                go.Scatter(
                    name = 'Average Training Precision',
                    x = epochs_range, 
                    y = precisions),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Average Validation Preicision',
                    x = epochs_range, 
                    y = val_precisions),
                row = 1, col = 1
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Average Training Recall',
                    x = epochs_range, 
                    y = recalls),
                row = 1, col = 2
            )
            fig.add_trace(
                go.Scatter(
                    name = 'Average Validation Recall',
                    x = epochs_range, 
                    y = val_recalls),
                row = 1, col = 2
            )
            fig.update_layout(title = 'Average Precision and Recall')
            fig.write_html(os.path.join(root_path, 'average_precision_and_recall.html'))
        else:
            self._logger.warn("Neither the model has been trained nor it has state.")

    def save_metadata(self):
        # Update new root path
        self._logger.info(f"Saving metadata in {self._root_path}")
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
        if timestamp is None:
            # get the lastest checkpoint provided that the timestamp is None
            max_datetime = None
            for f_name in glob.glob(os.path.join(".checkpoints", self._model_name, '*')):
                _,_, d_time = f_name.split('/')
                if max_datetime is None:
                    max_datetime = d_time
                elif datetime.strptime(max_datetime, SQL_TIMESTAMP) < datetime.strptime(d_time, SQL_TIMESTAMP):
                    max_datetime = d_time
        else:
            max_datetime = timestamp

        # load the model
        self._root_path = os.path.join('.checkpoints', self._model_name, max_datetime)
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

    def predict(self, x : Union[np.ndarray, str], result_dir = None, batch_size = 32):
        if isinstance(x, str):
            # glob files
            predicted_filenames = []
            predictionss = []
            image_predicted = 0
            for dirpath, _, filenames in os.walk(x):
                images = []
                for i, filename in enumerate(filenames):
                    loc = os.path.join(dirpath, filename)

                    try:
                        img = tf.keras.preprocessing.image.load_img(
                            loc, target_size=(self.img_height, self.img_width)
                        )
                        images.append(tf.keras.preprocessing.image.img_to_array(img))
                        predicted_filenames.append(filename)
                    except:
                        continue
                    if len(images) == batch_size: #predict by chunks
                        images = np.stack(images)
                        predictions = self.predict(images)
                        predictionss.append(predictions)
                        image_predicted += len(images)
                        images = []
                        self._logger.info(f'{image_predicted} images predicted')
                if len(images) > 0:
                    images = np.stack(images)
                    predictions = self.predict(images)
                    predictionss.append(predictions)
                    image_predicted += len(images)
                    images = []
                    self._logger.info(f'{image_predicted} images predicted')
            predictions = tf.concat(predictionss, axis = 0)
            #prepare for result directory
            if result_dir is not None:
                result = pd.DataFrame(columns = ['image_name', 'country'])
            for i, (fname, prediction) in enumerate(zip(predicted_filenames,predictions)):
                index = np.argmax(prediction)
                score = np.max(prediction)
                self._logger.info(f'{fname} : {self.class_names[index]} with confidence score of {round(score * 100, 2)}%')
                if result_dir is not None:
                    result.loc[i] = [fname,self.class_names[index]]

            if result_dir is not None:
                ensure_dir(result_dir)
                result.to_csv(os.path.join(result_dir, 'result.csv'), index = False)
        else:
            if x.ndim == 3:
                # expand dimension of one image to (1, img_height, img_width, 3)
                x = np.expand_dims(x, 0)
            if len(x) > batch_size:
                chunks = np.split(x)
                predictions = []
                for chunk in chunks:
                    p = self._model.predict(chunk)
                    predictions.append(p)
            else:
                predictions = self._model.predict(x)

        return predictions



