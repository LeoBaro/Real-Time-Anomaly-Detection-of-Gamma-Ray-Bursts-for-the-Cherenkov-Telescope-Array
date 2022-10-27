import numpy as np
import tensorflow as tf
from pathlib import Path

class CustomMSE(tf.keras.losses.Loss):
    """This MSE will weight the importance of the errors based on the energy of the photons.    
    """
    def __init__(self, n_features, name="custom_mse", weighted_average=True, output_dir="./"):
        super().__init__(name=name)
        self.mse_per_sample_features = None
        self.mse_per_sample = None
        self.output_dir = Path(output_dir)
        self.n_features = n_features
        self.weighted_average = weighted_average
        
    def get_weights(self):
        if self.weighted_average and self.n_features == 1:
            return tf.constant([1.0], dtype=tf.float32)
        elif self.weighted_average and self.n_features == 2:
            return tf.constant([2./3, 1./3], dtype=tf.float32)
        elif self.weighted_average and self.n_features == 3:
            return tf.constant([1./2, 1./3, 1./6], dtype=tf.float32)
        elif self.weighted_average and self.n_features == 4:
            return tf.constant([2./5, 3./10, 1./5, 1./10], dtype=tf.float32)
        elif not self.weighted_average and self.n_features == 1:
            return tf.constant([1.0], dtype=tf.float32)
        elif not self.weighted_average and self.n_features == 2:
            return tf.constant([1.0, 1.0], dtype=tf.float32)
        elif not self.weighted_average and self.n_features == 3:
            return tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        elif not self.weighted_average and self.n_features == 4:
            return tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        np.average: https://numpy.org/doc/stable/reference/generated/numpy.average.html

        y_true.shape = (batch_size, number_of_points, number_of_features)
        y_pred.shape = (batch_size, number_of_points, number_of_features)
        mse_per_sample_features = (batch_size, number_of_features)
        mse_per_sample = (batch_size,)
        """
        self.mse_per_sample_features = tf.math.reduce_mean(tf.math.square(y_true - y_pred), axis=1)
        
        self.mse_per_sample = tf.math.reduce_sum(self.mse_per_sample_features * self.get_weights(), axis=1) / tf.math.reduce_sum(self.get_weights())

        # self.mse_per_sample = np.average(self.mse_per_sample_features, axis=-1, weights=self.get_weights(self.mse_per_sample_features.shape[-1]))
        return tf.math.reduce_mean(self.mse_per_sample)

    
    def write_reconstruction_errors(self):
        if self.mse_per_sample is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir.joinpath("reconstruction_errors.csv"), "w") as ref:
                for err in self.mse_per_sample.numpy().squeeze():
                    ref.write(f"{err}\n")