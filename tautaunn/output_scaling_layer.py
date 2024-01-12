import tensorflow as tf


class CustomOutputScalingLayer(tf.keras.layers.Layer):
    def __init__(self, target_means, target_stds, **kwargs):
        """
        Constructor. Stores arguments as instance members.
        """

        super().__init__(**kwargs)

        self.target_means = target_means
        self.target_stds = target_stds

    def get_config(self):
        """
        Method that is required for model cloning and saving. It should return a
        mapping of instance member names to the actual members.
        """

        config = super().get_config()
        config["target_means"] = self.target_means
        config["target_stds"] = self.target_stds
        return config

    def call(self, features):
        """
        Payload of the layer that takes features and computes the requested output
        whose shape should match what is defined in compute_output_shape.
        """

        return features * self.target_stds + self.target_means
