from models.vgg16_grader_base import VGG16GraderBase
import tensorflow as tf

class VGG16GraderSurface(VGG16GraderBase):
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
        super(VGG16GraderSurface, self).__init__(
            'Surface',
            max_score,
            img_height,
            img_width,
            dim,
            learning_rate,
            epochs,
            clean_log,
            clean_checkpoints
        )

    def _define_meaty_layer(self, inputs):
        self._dense_0 = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01))(inputs)
        self._dense_1 = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01))(self._dense_0)
        self._dense_2 = tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01))(self._dense_1)
        self._dropout = tf.keras.layers.Dropout(0.3)(self._dense_2)
        return tf.keras.layers.Dense(1, activation = 'sigmoid')(self._dropout)