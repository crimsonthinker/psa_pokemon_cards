from utils.constants import SQL_TIMESTAMP
import numpy as np
import os
import tensorflow as tf
import datetime
import argparse

from models.unet import UNET
from utils.loader import UNETDataLoader

class UNETTrainer():
    def __init__(self, args):
        self.model = UNET((args.img_height, args.img_width, args.dim))
        self.dataloader = UNETDataLoader(
            batch_size = args.batch_size,
            original_height = args.original_height,
            original_width = args.original_width,
            dim = args.dim
        )
        self.model.build(input_shape=(1, args.img_height, args.img_width, args.dim))
        
        self.optimizer = tf.keras.optimizers.Adam(lr=5e-4)
        self.loss = tf.keras.losses.MeanAbsoluteError() # why this loss
        self.metric = tf.keras.metrics.BinaryAccuracy()
        self.epochs = args.epochs

        self.current_time = datetime.datetime.now().strftime(SQL_TIMESTAMP)
        self.saved_model_dir = os.path.join('.checkpoints', 'cropper', self.current_time)

        self.pretrained_model_path = os.path.join('.checkpoints', 'cropper', 'pretrained')

        self.val_ratio = args.val_ratio

    def train(self, from_pretrained = True):

        self.dataloader.load()
        self.dataloader.split(ratio = 1 - self.val_ratio)
        self.dataloader.shuffle()
        if from_pretrained: # if we start from pre-trained checkpoints
            self.model.load_weights(self.pretrained_model_path)

        for epoch in range(self.epochs):
            train_loss = []
            train_metric = []
            test_loss = []
            test_metric = []
            # train
            while(self.dataloader.is_enough_data(is_train = True)):
                inputs, ground_truths = self.dataloader.next_train_batch()
                loss, metric = self.gradient_descent(inputs, ground_truths)
                train_loss.append(loss.numpy())
                train_metric.append(metric.numpy())

            # make prediction for validation
            while(self.dataloader.isEnoughData(is_train = False)):
                inputs, ground_truths = self.dataloader.next_test_batch()
                loss, metric = self.predict(inputs, ground_truths, is_train=False)
                test_loss.append(loss.numpy())
                test_metric.append(metric.numpy())

            template = '>>> Epoch {}/{}, Train Loss: {:.4f}, Train Metric: {:.4f}, Test Loss: {:.4f}, Test Metric: {:.4f}'.format(
                                epoch + 1, self.nepochs, 
                                np.mean(train_loss), np.mean(train_metric), 
                                np.mean(test_loss), np.mean(test_metric))
            print(template)
            if epoch + 1 > self.checkpoint_since_epoch:
                self.save_weights(
                    epoch,
                    np.mean(train_loss), np.mean(train_metric), 
                    np.mean(test_loss), np.mean(test_metric)
                )
            self.dataloader.shuffle()
            self.dataloader.reset()

    @tf.function
    def gradient_descent(self, inputs, ground_truths):
        with tf.GradientTape() as tape:
            loss, metric = self.predict(
                inputs, ground_truths
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        return loss, metric

    @tf.function
    def predict(self, inputs, ground_truths, is_train=True):
        preds = self.model(inputs, training=is_train)

        loss = self.loss(preds, ground_truths)
        metric = self.metric(preds, ground_truths)
        return loss, metric

    def save_weights(
        self, epoch,
        train_loss, train_metric,
        test_loss, test_metric
    ):
        dir_name = f"{epoch + 1}_" \
                   + f"{train_loss:.5f}_{train_metric:.5f}_" \
                   + f"{test_loss:.5f}_{test_metric:.5f}"

        dir_path = os.path.join(self.saved_model_dir, dir_name)
        if not os.path.isdir(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)
        os.mkdir(dir_path)
        self.model.save_weights(dir_path + "/saved", save_format="tf")

    


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
    parser.add_argument("--epochs", type=int, default=15, nargs='?',
        help="Number of epochs for training session")
    parser.add_argument("--batch_size", type=int, default=32, nargs='?',
        help="Batch size for training session")
    parser.add_argument("--val_ratio", type=int, default=0.25, nargs='?',
        help="Ratio of validation data")
    args = parser.parse_args()

    # train u net for card segmentation
    trainer = UNETTrainer(args)
    trainer.model.summary()
    trainer.train(from_pretrained=False)