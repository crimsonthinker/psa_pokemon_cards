import numpy as np
import os
import random as rn
import tensorflow as tf
import datetime

from tensorflow.python.ops.variables import model_variables

from model import UNET
from dataloader import DataLoader

class Trainer():
    def __init__(self):
        self.model = UNET((512, 512, 3))
        self.dataloader = DataLoader()
        self.model.build(input_shape=(1, 512, 512, 3))
        self.nepochs = 200
        
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metric = tf.keras.metrics.BinaryAccuracy()

        self.current_time = datetime.datetime.now().strftime("%d_%m_%Y-%H%_M%_S")
        self.saved_model_dir = './checkpoints/cropper/{}/'.format(self.current_time)
        self.checkpoint_since_epoch = 150

    def train(self, isPretrained=True):
        self.dataloader.load()
        self.dataloader.slit(ratio=0.75)
        self.dataloader.shuffle()
        if isPretrained:
            self.model.load_weights(self.pretrained_model_path)

        for epoch in range(self.nepochs):
            train_loss = []
            train_metric = []
            test_loss = []
            test_metric = []
            while(self.dataloader.isEnoughData(isTrained=True)):
                inputs, ground_truths = self.dataloader.next_train_batch()
                loss, accurate = self.gradient_descent(inputs, ground_truths)
                train_loss.append(loss.numpy())
                train_metric.append(accurate.numpy())

            while(self.dataloader.isEnoughData(isTrained=False)):
                inputs, ground_truths = self.dataloader.next_test_batch()
                loss, accurate = self.predict(inputs, ground_truths, isTrained=False)
                test_loss.append(loss.numpy())
                test_metric.append(accurate.numpy())

            template = '>>> Epoch {}/{}, Train Loss: {}, Train Acccuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1, self.nepochs, 
                                sum(train_loss)/len(train_loss), sum(train_metric)/len(train_metric), 
                                sum(test_loss)/len(test_loss), sum(test_metric)/len(test_metric)))
            if epoch > self.checkpoint_since_epoch:
                self.save_weights(
                    epoch,
                    sum(train_loss)/len(train_loss), sum(train_metric)/len(train_metric),
                    sum(test_loss)/len(test_loss), sum(test_metric)/len(test_metric)
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
    def predict(self, inputs, ground_truths, isTrained=True):
        preds = self.model(inputs, training=isTrained)

        loss = self.loss(preds, ground_truths)
        metric = self.metric(preds, ground_truths)
        return loss, metric

    def save_weights(
        self, epoch,
        train_loss, train_accu,
        test_loss, test_accu
    ):
        dir_name = f"{epoch}_" \
                   + f"{train_loss:.4f}_{train_accu:.4f}_" \
                   + f"{test_loss:.4f}_{test_accu:.4f}"

        dir_path = f"{self.saved_model_dir}/{dir_name}"
        os.mkdir(dir_path)
        self.model.save_weights(dir_path + "/saved", save_format="tf")

    


if __name__ == '__main__':
    trainer = Trainer()
    trainer.model.summary()
    trainer.train(isPretrained=False)