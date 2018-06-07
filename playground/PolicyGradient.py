# coding=utf-8

import tensorflow as tf
import numpy as np
import gym


from base.model import BaseRLModel


class Agent(BaseRLModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(Agent, self).__init__(session, env, a_space, s_space, **options)

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()

        self.a_buffer, self.s_buffer, self.r_buffer = [], [], []

        self.session.run(tf.global_variables_initializer())

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.s_space])
            self.a = tf.placeholder(tf.int32,   [None, ])
            self.r = tf.placeholder(tf.float32, [None, ])

    def _init_nn(self, *args):
        with tf.variable_scope('actor_net'):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            # First dense.
            f_dense = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer)
            # Second dense.
            s_dense = tf.layers.dense(f_dense, 32, tf.nn.relu, kernel_initializer=w_initializer)
            # Action logits.
            self.a_logits = tf.layers.dense(s_dense, self.a_space, kernel_initializer=w_initializer)
            # Action prob.
            self.a_prob = tf.nn.softmax(self.a_logits)

    def _init_op(self):
        with tf.variable_scope('loss_func'):
            # one hot a.
            a_one_hot = tf.one_hot(self.a, self.a_space)
            # cross entropy.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a_one_hot, logits=self.a_logits)
            # loss func.
            self.loss_func = tf.reduce_mean(cross_entropy * self.r)
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)

    def predict(self, s):
        a_prob = self.session.run(self.a_prob, {self.s: [s]})
        return np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

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
        # Minimize loss.
        _, loss = self.session.run([self.optimizer, self.loss_func], {
            self.s: self.s_buffer,
            self.a: self.a_buffer,
            self.r: self.r_buffer
        })
        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []

    def run(self):
        if self.mode == 'train':
            for episode in range(self.train_episodes):
                s, r_episode = self.env.reset(), 0
                while True:
                    if episode > 400:
                        self.env.render()
                    a = self.predict(s)
                    s_n, r, done, _ = self.env.step(a)
                    if done:
                        r = -5
                    r_episode += r
                    self.snapshot(s, a, r, s_n)
                    s = s_n
                    if done:
                        break
                self.train()
                if episode % 50 == 0:
                    self.logger.warning('Episode: {} | Rewards: {}'.format(episode, r_episode))
                    self.save()
        else:
            for episode in range(self.eval_episodes):
                s, r_episode = self.env.reset()
                while True:
                    a = self.predict(s)
                    s_n, r, done, _ = self.env.step(a)
                    r_episode += r
                    s = s_n
                    if done:
                        break


def main(_):
    # Make env.
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    # Init session.
    session = tf.Session()
    # Init agent.
    agent = Agent(session, env, env.action_space.n, env.observation_space.shape[0], **{
        'model_name': 'PolicyGradient',
    })
    agent.run()


if __name__ == '__main__':
    tf.app.run()
