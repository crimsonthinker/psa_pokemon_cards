from utils.utilities import ensure_dir
from utils.constants import SQL_TIMESTAMP
import numpy as np
import os
import tensorflow as tf
import datetime
import argparse
import json

from models.unet import UNET
from utils.loader import UNETDataLoader

class UNETTrainer():
    def __init__(self, args):
        self.model = UNET((args.img_height, args.img_width, args.dim))
        self.dataloader = UNETDataLoader(
            batch_size = args.batch_size,
            original_height = args.origin_img_height,
            original_width = args.origin_img_width,
            dim = args.dim
        )
        self.model.build(input_shape=(1, args.img_height, args.img_width, args.dim))
        
        self.optimizer = tf.keras.optimizers.Adam(lr=5e-5)
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

        self.epochs = args.epochs

        self.current_time = datetime.datetime.now().strftime(SQL_TIMESTAMP)
        self.saved_model_dir = os.path.join('checkpoint', 'cropper', self.current_time)

        self.pretrained_model_path = os.path.join('checkpoint/cropper/pretrained', '2021-08-27 14:47:32')

        self.val_ratio = args.val_ratio

    def train(self, from_pretrained = True):
        self.history = []
        self.dataloader.load()
        self.dataloader.split(ratio = 1 - self.val_ratio)
        self.dataloader.shuffle()
        if from_pretrained: # if we start from pre-trained checkpoints
            self.model.load_weights(self.pretrained_model_path)

        for epoch in range(self.epochs):
            train_loss = []
            train_accuracy = []
            train_iou = []
            test_loss = []
            test_accuracy = []
            test_iou = []
            # train
            while(self.dataloader.is_enough_data(is_train = True)):
                inputs, ground_truths = self.dataloader.next_train_batch()
                loss, accuracy, iou = self.gradient_descent(inputs, ground_truths)
                train_loss.append(loss.numpy())
                train_accuracy.append(accuracy.numpy())
                train_iou.append(iou.numpy())

            # make prediction for validation
            while(self.dataloader.is_enough_data(is_train = False)):
                inputs, ground_truths = self.dataloader.next_test_batch()
                loss, accuracy, iou = self.predict(inputs, ground_truths, is_train=False)
                test_loss.append(loss.numpy())
                test_accuracy.append(accuracy.numpy())
                test_iou.append(iou.numpy())

            self.history.append(
                {
                    'epoch' : epoch,
                    'train_loss' : float(np.mean(train_loss)),
                    'test_loss' : float(np.mean(test_loss)),
                    'train_accuracy' : float(np.mean(train_accuracy)),
                    'test_accuracy' : float(np.mean(test_accuracy)),
                    'train_iou' : float(np.mean(train_iou)),
                    'test_iou' : float(np.mean(test_iou))
                })

            self.save_weights(epoch, datetime.datetime.now().strftime(SQL_TIMESTAMP))
            template = '>>> Epoch {}\n Train Loss: {:.4f}, Train Accuracy: {:.4f}, Train IoU: {:.4f}\n Test Loss: {:.4f}, Test Accuracy: {:.4f}, Test IoU: {:.4f}'.format(
                                epoch + 1, 
                                np.mean(train_loss), np.mean(train_accuracy), np.mean(train_iou), 
                                np.mean(test_loss), np.mean(test_accuracy), np.mean(test_iou)
            )
            print(template)
            self.dataloader.shuffle()
            self.dataloader.reset()

        with open(os.path.join(self.saved_model_dir, 'result.json'), 'w') as f:
            json.dump(self.history, f)

    @tf.function
    def gradient_descent(self, inputs, ground_truths):
        with tf.GradientTape() as tape:
            loss, accuracy, iou = self.predict(
                inputs, ground_truths
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        return loss, accuracy, iou

    @tf.function
    def predict(self, inputs, ground_truths, is_train=True):
        preds = self.model(inputs, training=is_train)

        loss = self.loss(preds, ground_truths)
        accuracy = self.accuracy_metric(preds, ground_truths)
        iou = self.iou_metric(preds, ground_truths)
        return loss, accuracy, iou

    def save_weights(self, epoch, name):
        dir_path = os.path.join(self.saved_model_dir, str(epoch + 1))
        if not os.path.isdir(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)
        os.mkdir(dir_path)
        self.model.save_weights(dir_path + '/{}'.format(name), save_format="tf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_img_height", type=int, default=2698, nargs='?',
        help="Origin image height for the training session")
    parser.add_argument("--origin_img_width", type=int, default=1620, nargs='?',
        help="Original image width for the training session")
    parser.add_argument("--img_height", type=int, default=512, nargs='?',
        help="Image height for the training session")
    parser.add_argument("--img_width", type=int, default=512, nargs='?',
        help="Image width for the training session")
    parser.add_argument("--dim", type=int, default=3, nargs='?',
        help="Image didmension for the training session")
    parser.add_argument("--epochs", type=int, default=40, nargs='?',
        help="Number of epochs for training session")
    parser.add_argument("--batch_size", type=int, default=32, nargs='?',
        help="Batch size for training session")
    parser.add_argument("--val_ratio", type=int, default=0.25, nargs='?',
        help="Ratio of validation data")
    args = parser.parse_args()

    # train u net for card segmentation
    trainer = UNETTrainer(args)
    trainer.model.summary()
    trainer.train(from_pretrained=True)