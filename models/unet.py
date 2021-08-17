import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout

from utils.constants import SMOOTH

class UNET(tf.keras.Model):
    def __init__(self, input_size=(512, 512, 3)):
        super().__init__()

        self.norm = tf.keras.layers.BatchNormalization()
        self.dropout = Dropout(0.5)

        self.conv1_1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv2_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
    
        self.conv3_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv4_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv5_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')

        self.tconv6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.conv6_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv6_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')

        self.tconv7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
        self.conv7_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv7_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')

        self.tconv8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
        self.conv8_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.conv8_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')

        self.tconv9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')
        self.conv9_1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.conv9_2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        
        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        # 512 * 512 * 3
        s = self.norm(inputs)
        s = self.dropout(s)
        # 512 * 512 * 3
        c1 = self.conv1_1(s)
        c1 = self.conv1_2(c1)
        # 512 * 512 * 8
        p1 = self.pool1(c1)
        # 256 * 256 * 8

        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        # 256 * 256 * 16
        p2 = self.pool2(c2)
        # 128 * 128 * 16

        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        # 128 * 128 * 32
        p3 = self.pool3(c3)
        # 64 * 64 * 32

        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        # 64 * 64 * 64
        p4 = self.pool4(c4)
        # 32 * 32 * 64

        c5 = self.conv5_1(p4)
        c5 = self.conv5_2(c5)
        # 32 * 32 * 128

        u6 = self.tconv6(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = self.conv6_1(u6)
        c6 = self.conv6_2(c6)

        u7 = self.tconv7(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = self.conv7_1(u7)
        c7 = self.conv7_2(c7)

        u8 = self.tconv8(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = self.conv8_1(u8)
        c8 = self.conv8_2(c8)

        u9 = self.tconv9(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = self.conv9_1(u9)
        c9 = self.conv9_2(c9)

        return self.outputs(c9)

