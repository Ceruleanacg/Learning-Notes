# coding=utf-8

import tensorflow as tf

from abc import abstractmethod
from utility.logger import *
from static import *

KEY_TRAIN_EPISODE = 'train_episodes'
KEY_LEARNING_RATE = 'learning_rate'
KEY_SAVE_EPISODE = 'save_episode'
KEY_EVAL_EPISODE = 'eval_episode'
KEY_BUFFER_SIZE = 'buffer_size'
KEY_MODEL_NAME = 'model_name'
KEY_BATCH_SIZE = 'batch_size'
KET_EPSILON = 'epsilon'
KEY_GAMMA = 'gamma'
KEY_MODE = 'mode'
KEY_TAU = 'tau'


class BaseModel(object):

    def __init__(self, session, env, **options):
        # Init session.
        self.session = session
        # Init env.
        self.env = env
        # Init options.
        self._init_options(options)

    def _init_saver(self):
        self.checkpoint_path = os.path.join(CKPT_DIR, self.model_name, 'ckpt')
        self.saver = tf.train.Saver()

    def _init_summary_writer(self):
        self.summary_path = os.path.join(SUMR_DIR, self.model_name, DATETIME_NOW)
        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, self.session.graph)

    def _init_options(self, options):

        try:
            self.model_name = options[KEY_MODEL_NAME]
        except KeyError:
            self.model_name = 'model'

        try:
            self.mode = options[KEY_MODE]
        except KeyError:
            self.mode = 'train'

        try:
            self.learning_rate = options[KEY_LEARNING_RATE]
        except KeyError:
            self.learning_rate = 0.003

        try:
            self.batch_size = options[KEY_BATCH_SIZE]
        except KeyError:
            self.batch_size = 64

        self.logger = generate_model_logger(self.model_name)

    def save(self):
        self.saver.save(self.session, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.session, self.checkpoint_path)


class BaseRLModel(BaseModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(BaseRLModel, self).__init__(session, env, **options)
        # Init spaces.
        self.a_space, self.s_space = a_space, s_space
        # Init buffer count.
        self.buffer_count = 0

    def _init_options(self, options):
        super(BaseRLModel, self)._init_options(options)

        try:
            self.train_episodes = options[KEY_TRAIN_EPISODE]
        except KeyError:
            self.train_episodes = 1000

        try:
            self.eval_episodes = options[KEY_EVAL_EPISODE]
        except KeyError:
            self.eval_episodes = 30

        try:
            self.gamma = options[KEY_GAMMA]
        except KeyError:
            self.gamma = 0.95

        try:
            self.tau = options[KEY_TAU]
        except KeyError:
            self.tau = 0.01

        try:
            self.epsilon = options[KET_EPSILON]
        except KeyError:
            self.epsilon = 0.9

        try:
            self.buffer_size = options[KEY_BUFFER_SIZE]
        except KeyError:
            self.buffer_size = 10000

        try:
            self.save_episode = options[KEY_SAVE_EPISODE]
        except KeyError:
            self.save_episode = 50

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, s):
        pass

    @abstractmethod
    def snapshot(self, s, a, r, s_n):
        pass
