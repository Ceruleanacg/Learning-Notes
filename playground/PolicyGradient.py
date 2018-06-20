# coding=utf-8

import numpy as np
import gym

from base.model import *
from utility.launcher import start_game


class Agent(BaseRLModel):

    def __init__(self, a_space, s_space, **options):
        super(Agent, self).__init__(a_space, s_space, **options)

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

        self.a_buffer, self.s_buffer, self.r_buffer = [], [], []

        self.session.run(tf.global_variables_initializer())

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.s_space])
            self.a = tf.placeholder(tf.int32,   [None, ])
            self.r = tf.placeholder(tf.float32, [None, ])
            # Add summary.
            tf.summary.histogram('rewards', self.r)

    def _init_nn(self, *args):
        with tf.variable_scope('actor_net'):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            # First dense.
            f_dense = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=w_initializer)
            # Second dense.
            s_dense = tf.layers.dense(f_dense, 64, tf.nn.relu, kernel_initializer=w_initializer)
            # Action logits.
            self.a_logits = tf.layers.dense(s_dense, self.a_space, kernel_initializer=w_initializer)
            # Action prob.Î
            self.a_prob = tf.nn.softmax(self.a_logits)

    def _init_op(self):
        with tf.variable_scope('loss_func'):
            # one hot a.
            a_one_hot = tf.one_hot(self.a, self.a_space)
            # cross entropy.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a_one_hot, logits=self.a_logits)
            # loss func.
            self.loss_func = tf.reduce_mean(cross_entropy * self.r)
            # add summary.
            tf.summary.scalar('r_cross_entropy', self.loss_func)
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(initial_value=0)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)

    def predict(self, s):
        a_prob = self.session.run(self.a_prob, {self.s: [s]})
        if self.mode == 'train':
            return np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())
        else:
            return np.argmax(a_prob)

    def snapshot(self, s, a, r, _):
        self.a_buffer.append(a)
        self.s_buffer.append(s)
        self.r_buffer.append(r)

    def train(self):
        # Copy r_buffer
        r_buffer = self.r_buffer
        # Init r_tau
        r_tau = 0
        # Calculate r_tau
        for index in reversed(range(0, len(r_buffer))):
            r_tau = r_tau * self.gamma + r_buffer[index]
            self.r_buffer[index] = r_tau
        # Make ops.
        ops = [self.optimizer, self.loss_func]
        if self.training_step % 5 == 0:
            ops.append(self.merged_summary_op)
        # Minimize loss.
        results = self.session.run(ops, {
            self.s: self.s_buffer,
            self.a: self.a_buffer,
            self.r: self.r_buffer
        })

        if self.training_step % 10 == 0:
            self.summary_writer.add_summary(results[-1], global_step=self.training_step)

        self.training_step += 1

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []


def main(_):
    # Make env.
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    # Init agent.
    agent = Agent(env.action_space.n, env.observation_space.shape[0], **{
        KEY_MODEL_NAME: 'PolicyGradient',
        KEY_TRAIN_EPISODE: 10000
    })
    start_game(env, agent)


if __name__ == '__main__':
    tf.app.run()
