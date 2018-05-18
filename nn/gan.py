import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class GAN(object):

    def __init__(self, x_width, x_height, x_channel, **options):

        self.x_width = x_width
        self.x_height = x_height
        self.x_channel = x_channel

        # TODO - GPU limit.
        self.session = tf.Session()

    def _init_input(self):
        self.x_input = tf.placeholder(tf.float32, [None, self.x_height, self.x_width, self.x_channel])
        self.y_output = tf.placeholder(tf.float32, [None, self.x_height, self.x_width, self.x_channel])

    def _init_nn(self):
        with tf.variable_scope('generator'):
            # Reshape x.
            x = tf.reshape(self.x_input, [-1, self.x_height * self.x_width * self.x_channel])
            # 1-Dense.
            dense = tf.layers.dense(x, activation=tf.nn.relu)
            pass

    def _init_op(self):
        pass

    def train(self):
        pass
