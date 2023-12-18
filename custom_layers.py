# encoding: utf-8

import tensorflow as tf


class CustomEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, expected_inputs, **kwargs):
        """
        Constructor. Stores arguments as instance members.
        """
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.expected_inputs = expected_inputs
        self.offsets = None
        self.embedding_layer = None
        self.flatten_layer = None

    def get_config(self):
        """
        Method that is required for model cloning and saving. It should return a
        mapping of instance member names to the actual members.
        """
        config = super().get_config()
        config["output_dim"] = self.output_dim
        config["expected_inputs"] = self.expected_inputs
        return config

    def compute_output_shape(self, input_shape):
        """
        Method that, given an input shape, defines the shape of the output tensor.
        This way, the entire model can be built without actually calling it.
        """
        return (input_shape[0], input_shape[1] * self.output_dim)

    def build(self, input_shape):
        """
        Any variables defined by this layer should be created inside this method.
        This helps Keras to defer variable registration to the point where it is
        needed the first time, and in particular not at definition time.
        """
        self.offsets = [
            sum(len(x) for x in self.expected_inputs[:i])
            for i in range(len(self.expected_inputs))
        ]

        self.embedding_layer = tf.keras.layers.Embedding(
            len(sum(self.expected_inputs, [])),
            self.output_dim,
            input_length=len(self.expected_inputs),
        )

        self.flatten_layer = tf.keras.layers.Flatten()

        super().build(input_shape)

    def call(self, input_features):
        """
        Payload of the layer that takes features and computes the requested output
        whose shape should match what is defined in compute_output_shape.
        """
        indices = []
        # TODO: keras has no clue about the input shape at this point, and input_features.shape cannot be used since
        # a (None, None) shaped tracer is send through the call methods that crashes everything -.-
        # for i in range(input_features.shape[1]):
        for i in range(6):
            var = input_features[:, i]
            indices.append(tf.where(var[..., None] == self.expected_inputs[i])[:, 1] + self.offsets[i])

        embed_input = tf.concat(
            [i[..., None] for i in indices],
            axis=-1,
        )
        embed_output = self.embedding_layer(embed_input)

        return self.flatten_layer(embed_output)


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

    def call(self, input_features):
        """
        Payload of the layer that takes features and computes the requested output
        whose shape should match what is defined in compute_output_shape.
        """
        return input_features * self.target_stds + self.target_means
