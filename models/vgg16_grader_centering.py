from models.vgg16_grader_base import VGG16GraderBase
import tensorflow as tf

class VGG16GraderCentering(VGG16GraderBase):
    def __init__(
        self,        
        max_score : int = 10,
        img_height : int = 256, 
        img_width : int = 256, 
        dim : int = 3,
        learning_rate : float = 0.001,
        epochs : int = 10,
        clean_log : bool = False,
        clean_checkpoints : bool = False):
        super(VGG16GraderCentering, self).__init__(
            'Centering',
            max_score,
            img_height,
            img_width,
            dim,
            learning_rate,
            epochs,
            clean_log,
            clean_checkpoints
        )

    def _define_meaty_layer(self):
        self._layer_only = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, 
                activation = 'relu',
                kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(64, 
                activation = 'relu',
                kernel_regularizer = tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ], name = 'meaty_layer')