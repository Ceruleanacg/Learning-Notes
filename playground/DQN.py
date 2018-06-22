# coding=utf-8

import numpy as np
import gym

from base.model import *
from utility.launcher import start_game

tf.set_random_seed(7)


class Agent(BaseRLModel):

    def __init__(self, a_space, s_space, **options):
        super(Agent, self).__init__(a_space, s_space, **options)

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()

        self.buffer = np.zeros((self.buffer_size, self.s_space + 1 + 1 + self.s_space))
        self.buffer_count = 0

        self.update_target_net_step = 200

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s_n = tf.placeholder(tf.float32, [None, self.s_space])
            self.s = tf.placeholder(tf.float32,   [None, self.s_space])
            self.r = tf.placeholder(tf.float32,   [None, ])
            self.a = tf.placeholder(tf.int32,     [None, ])

    def _init_nn(self, *args):
        # w,b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.00003)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('predict_q_net'):
            phi_state = tf.layers.dense(self.s,
                                        64,
                                        tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer)

            self.q_predict = tf.layers.dense(phi_state,
                                             self.a_space,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer)

        with tf.variable_scope('target_q_net'):
            phi_state_next = tf.layers.dense(self.s_n,
                                             64,
                                             tf.nn.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer)

            self.q_target = tf.layers.dense(phi_state_next,
                                            self.a_space,
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer)

    def _init_op(self):
        with tf.variable_scope('q_real'):
            # size of q_value_real is [BATCH_SIZE, 1]
            max_q_value = tf.reduce_max(self.q_target, axis=1)
            q_next = self.r + self.gamma * max_q_value
            self.q_next = tf.stop_gradient(q_next)

        with tf.variable_scope('q_predict'):
            # size of q_value_predict is [BATCH_SIZE, 1]
            action_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval = tf.gather_nd(self.q_predict, action_indices)

        with tf.variable_scope('loss'):
            self.loss_func = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_eval, name='mse'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)

        with tf.variable_scope('update_target_net'):
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_net')
            p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict_q_net')
            self.update_q_net = [tf.assign(t, e) for t, e in zip(t_params, p_params)]

        self.session.run(tf.global_variables_initializer())

    def predict(self, s):
        if np.random.uniform() < self.epsilon:
            a = np.argmax(self.session.run(self.q_predict, feed_dict={self.s: s[np.newaxis, :]}))
        else:
            a = np.random.randint(0, self.a_space)
        return a

    def snapshot(self, s, a, r, s_n):
        self.buffer[self.buffer_count % self.buffer_size, :] = np.hstack((s, [a, r], s_n))
        self.buffer_count += 1

    def train(self):
        for train_step in range(self.train_steps):
            # Update target net if need.
            if self.training_step % self.update_target_net_step == 0:
                self.session.run(self.update_q_net)
            # Get batch.
            if self.buffer_count < self.batch_size:
                batch = self.buffer[np.random.choice(self.buffer_count, size=self.batch_size), :]
            else:
                batch = self.buffer[np.random.choice(self.buffer_size, size=self.batch_size), :]

            s = batch[:, :self.s_space]
            s_n = batch[:, -self.s_space:]
            a = batch[:, self.s_space].reshape((-1))
            r = batch[:, self.s_space + 1]

            _, cost = self.session.run([self.train_op, self.loss_func], {
                self.s: s, self.a: a, self.r: r * 5, self.s_n: s_n
            })

            self.training_step += 1


def main(_):
    # Make env.
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    # Init session.
    # Init agent.
    agent = Agent(env.action_space.n, env.observation_space.shape[0], **{
        KEY_MODEL_NAME: 'DQN',
        KEY_TRAIN_EPISODE: 500
    })
    start_game(env, agent)


if __name__ == '__main__':
    tf.app.run()
