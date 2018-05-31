import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym


class Agent(object):

    def __init__(self, a_space, s_space, **options):

        self.session = tf.Session()

        self.a_space, self.s_space = a_space, s_space

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []

        self._init_options(options)
        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.s = tf.placeholder(tf.float32, [None, self.s_space])
        self.r = tf.placeholder(tf.float32, [None, ])
        self.a = tf.placeholder(tf.int32, [None, ])

    def _init_nn(self):
        # Kernel init.
        w_init = tf.random_normal_initializer(.0, .3)
        # Dense 1.
        dense_1 = tf.layers.dense(self.s,
                                  32,
                                  tf.nn.relu,
                                  kernel_initializer=w_init)
        # Dense 2.
        dense_2 = tf.layers.dense(dense_1,
                                  32,
                                  tf.nn.relu,
                                  kernel_initializer=w_init)
        # Action logits.
        self.a_logits = tf.layers.dense(dense_2,
                                        self.a_space,
                                        kernel_initializer=w_init)
        # Action prob.
        self.a_prob = tf.nn.softmax(self.a_logits)

    def _init_op(self):
        # One hot action.
        action_one_hot = tf.one_hot(self.a, self.a_space)
        # Calculate cross entropy.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=action_one_hot, logits=self.a_logits)
        self.loss_func = tf.reduce_mean(cross_entropy * self.r)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_func)
        self.session.run(tf.global_variables_initializer())

    def _init_options(self, options):

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.95

    def predict(self, state):
        action_prob = self.session.run(self.a_prob, feed_dict={self.s: state[np.newaxis, :]})
        return np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel())

    def save_transition(self, state, action, reward):
        self.s_buffer.append(state)
        self.a_buffer.append(action)
        self.r_buffer.append(reward)

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
        _, loss = self.session.run([self.train_op, self.loss_func], feed_dict={
            self.s: self.s_buffer,
            self.a: self.a_buffer,
            self.r: self.r_buffer,
        })

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []


def main(_):

    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped

    model = Agent(env.action_space.n, env.observation_space.shape[0])

    r_sum_list, r_episode_sum = [], None

    for episode in range(500):
        # Reset env.
        s, r_episode = env.reset(), 0
        # Start episode.
        while True:
            # if episode > 80:
            #     env.render()
            # Predict action.
            a = model.predict(s)
            # Iteration.
            s_n, r, done, _ = env.step(a)
            if done:
                r = -5
            r_episode += r
            # Save transition.
            model.save_transition(s, a, r)
            s = s_n
            if done:
                if r_episode_sum is None:
                    r_episode_sum = sum(model.r_buffer)
                else:
                    r_episode_sum = r_episode_sum * 0.99 + sum(model.r_buffer) * 0.01
                r_sum_list.append(r_episode_sum)
                break
        # Start train.
        model.train()
        if episode % 50 == 0:
            print("Episode: {} | Reward is: {}".format(episode, r_episode))

    plt.plot(np.arange(len(r_sum_list)), r_sum_list)
    plt.title('Actor Only on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == '__main__':
    tf.app.run()
