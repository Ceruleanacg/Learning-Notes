import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from base.model import *


class Agent(BaseSLModel):

    def __init__(self, x_space, y_space, x_train, y_train, x_test, y_test, **options):
        super(Agent, self).__init__(x_space, y_space, x_train, y_train, x_test, y_test, **options)

        self._init_options(options)
        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

        self.session.run(tf.global_variables_initializer())

    def _init_input(self, *args):
        self.x_input = tf.placeholder(tf.float32, [None, self.x_space])
        self.y_input = tf.placeholder(tf.float32, [None, self.y_space])

    def _init_nn(self, *args):
        with tf.variable_scope('MLP'):
            f_dense = tf.layers.dense(self.x_input, 32, tf.nn.relu)
            s_dense = tf.layers.dense(f_dense, 32, tf.nn.relu)
            y_predict = tf.layers.dense(s_dense, self.y_space)
            self.y_predict = y_predict

    def _init_op(self):
        with tf.variable_scope('loss_func'):
            # self.loss_func = tf.reduce_mean(tf.square(self.y_input - self.y_predict) * tf.abs(self.y_predict))
            # self.loss_func = tf.reduce_mean(tf.square(self.y_input - self.y_predict) * tf.square(self.y_input))
            self.loss_func = tf.losses.mean_squared_error(self.y_input, self.y_predict)
            tf.summary.scalar('mse', self.loss_func)
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)

    def train(self):
        # Get data size.
        data_size = len(self.x_train)
        for train_step in range(30000):
            # Get mini batch.
            # indices = np.random.choice(data_size, size=self.batch_size)
            # x_batch = self.x_train[indices]
            # y_batch = self.y_train[indices]
            x_batch = self.x_train
            y_batch = self.y_train
            # Train op.
            ops = [self.optimizer, self.loss_func]
            if train_step % 500 == 0:
                ops.append(self.merged_summary_op)
            # Train.
            results = self.session.run(ops, {
                self.x_input: x_batch,
                self.y_input: y_batch,
            })
            # Add summary.
            if train_step % 500 == 0:
                self.summary_writer.add_summary(results[-1], global_step=self.training_step)
            # Log loss.
            if train_step % 10 == 0:
                self.save()
                self.logger.warning('Step: {0}, Training loss: {1:.10f}'.format(train_step, results[1]))
                self.evaluate()
            self.training_step += 1

    def predict(self, s):
        y_predict = self.session.run(self.y_predict, {self.x_input: s})
        return y_predict

    def evaluate(self):
        y_predict, loss = self.session.run([self.y_predict, self.loss_func], {
            self.x_input: self.x_test,
            self.y_input: self.y_test
        })

        self.logger.warning('Step: {0}, Testing loss: {1:.10f}'.format(self.training_step, loss))


if __name__ == '__main__':

    x_train = np.linspace(-np.pi, np.pi, num=200).reshape((-1, 1)) + np.random.normal()
    y_train = np.sin(x_train)

    x_test = np.linspace(-np.pi, np.pi, num=50).reshape((-1, 1))
    y_test = np.sin(x_test)

    agent = Agent(x_train[0].shape[0],
                  1,
                  x_train,
                  y_train,
                  x_test,
                  y_test)

    agent.train()
