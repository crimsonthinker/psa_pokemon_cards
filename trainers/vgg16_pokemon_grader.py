import numpy as np
import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date, datetime
import os
import shutil
import glob
import pickle
import pandas as pd
from typing import Union

from tensorflow.keras.applications import VGG16

from utils.utilities import *
from utils.constants import *
from utils.loader import *

class VGG16PokemonGrader(object):
    def __init__(self, 
        class_names = [], 
        img_height = 256, 
        img_width = 256, 
        dim = 3,
        learning_rate = 0.001,
        epochs = 10,
        clean_log = False,
        clean_checkpoints = False):

        self._logger = get_logger("VGG16PokemonGrader")
        self._model_name = 'vgg16_pokemon_grader'

        # remove log folder
        if os.path.exists('.log') and clean_log:
            shutil.rmtree('.log')
        elif not os.path.exists('.log'):
            ensure_dir('.log')

        # clean checkpoints
        # clean checkpoints
        if os.path.exists(os.path.join('.checkpoints', self._model_name)) and clean_checkpoints:
            shutil.rmtree(os.path.join('.checkpoints', self._model_name))
        elif not os.path.exists(os.path.join('.checkpoints', self._model_name)):
            ensure_dir(os.path.join('.checkpoints', self._model_name))
        
        self.class_names = class_names

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
        for layer in self._base_model.layers:
            layer.trainable = False

        if img_height is not None and img_width is not None:
            self._layer_only = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, 
                    activation = 'relu',
                    kernel_regularizer = tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.class_names), activation = 'softmax')
            ], name = 'temple_head_part')
        elif img_height is None and img_width is None:
            self._layer_only = tf.keras.Sequential([], name = 'core_network')
        else:
            raise ValueError("Both of img_height and img_width must be either None or have values")

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
            loss = 'categorical_crossentropy',
            metrics =
                ['accuracy'] + \
                [tf.keras.metrics.Precision(class_id = x, name = f'precision_{name}') for x, name in enumerate(self.class_names)] + \
                [tf.keras.metrics.Recall(class_id = x, name = f'recall_{name}') for x, name in enumerate(self.class_names)] + \
                [
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives(),
                    tf.keras.metrics.TrueNegatives()
                ]
        )

        
    def get_summary(self):
        self._model.summary()

    def set_epoch(self, new_epoch : int):
        self._epochs = new_epoch

    def train_and_evaluate(self, dataset : ImageLoader):
        try:
            self._logger.info(f"""
                Traning TempleClassfier:
                    Image resolution (hxw): {self.img_height}x{self.img_width}
                    Number of classes: {len(self.class_names)}
                    Number of epochs: {self._epochs}
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
        except:
            self._logger.info("Interrupt training session")

    def visualize_evaluation(self, visualize_result = False, save_as_html = False):
        if self._history is not None and (visualize_result or save_as_html):
            self._logger.info("Visualizing results")
            epochs_range = list(range(self._epochs))

            if save_as_html:
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
            if visualize_result:
                fig.show()
            if save_as_html:
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
            if visualize_result:
                fig.show()
            if save_as_html:
                fig.write_html(os.path.join(root_path, 'average_precision_and_recall.html'))

            # class wise precision and recall
            for class_name in self.class_names:
                fig = make_subplots(
                    rows = 1,
                    cols = 2
                )
                fig.add_trace(
                    go.Scatter(
                        name = f'Training Precision of {class_name}',
                        x = epochs_range, 
                        y = self._history.history[f'precision_{class_name}']),
                    row = 1, col = 1
                )
                fig.add_trace(
                    go.Scatter(
                        name = f'Validation Precision of {class_name}',
                        x = epochs_range, 
                        y = self._history.history[f'val_precision_{class_name}']),
                    row = 1, col = 1
                )
                fig.add_trace(
                    go.Scatter(
                        name = f'Training Recall of {class_name}',
                        x = epochs_range, 
                        y = self._history.history[f'recall_{class_name}']),
                    row = 1, col = 2
                )
                fig.add_trace(
                    go.Scatter(
                        name = f'Validation Recall of {class_name}',
                        x = epochs_range, 
                        y = self._history.history[f'val_precision_{class_name}']),
                    row = 1, col = 2
                )
                fig.update_layout(title = f'Precision and Recall of class {class_name}')
                if visualize_result:
                    fig.show()
                if save_as_html:
                    fig.write_html(os.path.join(root_path, f'precision_and_recall_{class_name}.html'))
        else:
            self._logger.warn("Neither the model has been trained nor it has state.")

    def save(self):
        ensure_dir(self._root_path)
        self._logger.info(f"Saving the models and class names in {self._root_path}")
        # save whole model with data augmentation
        self._model.save(self._root_path)
        # save class names as pickle
        with open(os.path.join(self._root_path, 'classes.pkl'), 'wb') as f:
            pickle.dump(self.class_names, f)


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
        self._model = tf.keras.models.load_model(self._root_path)

        _, self.img_height, self.img_width, self.dim = self._model.layers[0].output_shape

        # load the class names
        with open(os.path.join(self._root_path, 'classes.pkl'), 'rb') as f:
            self.class_names = pickle.load(f)

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
            if x.ndim == 3: #one image only
                x = np.expand_dims(x, 0)
            predictions = self._model.predict(x)

        return predictions



