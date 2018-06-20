# coding=utf-8

import tensorflow as tf

from abc import abstractmethod
from utility.logger import *
from static import *

KEY_TRAIN_EPISODE = 'train_episodes'
KEY_LEARNING_RATE = 'learning_rate'
KEY_ENABLE_EAGER = 'enable_eager'
KEY_SAVE_EPISODE = 'save_episode'
KEY_EVAL_EPISODE = 'eval_episode'
KEY_BUFFER_SIZE = 'buffer_size'
KEY_TRAIN_STEPS = 'train_steps'
KEY_MODEL_NAME = 'model_name'
KEY_BATCH_SIZE = 'batch_size'
KEY_SEQ_LENGTH = 'seq_length'
KEY_SAVE_DIR = 'save_dir'
KEY_SESSION = 'session'
KET_EPSILON = 'epsilon'
KEY_GAMMA = 'gamma'
KEY_MODE = 'mode'
KEY_TAU = 'tau'


class BaseModel(object):

    def __init__(self, **options):
        # Init vars.
        self.mode = 'train'
        self.save_dir = None
        self.training_step = 0
        self.checkpoint_path = None
        # Init parameters.
        self._init_options(options)
        self._init_logger()

    def _init_logger(self):
        self.logger = get_logger(self.model_name, self.mode, 'algo')

    def _init_saver(self):
        save_dir = os.path.join(CHECKPOINTS_DIR, self.model_name, self.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.checkpoint_path = os.path.join(CHECKPOINTS_DIR, self.model_name, self.save_dir, 'ckpt')
        self.saver = tf.train.Saver()

    def _init_summary_writer(self):
        self.summary_path = os.path.join(SUMMARIES_DIR, self.model_name, self.save_dir, DATETIME_NOW)
        self.summary_writer = tf.summary.FileWriter(self.summary_path, graph=self.session.graph)
        self.merged_summary_op = tf.summary.merge_all()

    def _init_options(self, options):

        try:
            self.enable_eager = options[KEY_ENABLE_EAGER]
        except KeyError:
            self.enable_eager = False

        try:
            self.session = options[KEY_SESSION]
        except KeyError:
            # config = tf.ConfigProto(device_count={"CPU": 1},
            #                         inter_op_parallelism_threads=1,
            #                         intra_op_parallelism_threads=8,
            #                         log_device_placement=True)
            self.session = tf.Session()

        try:
            self.model_name = options[KEY_MODEL_NAME]
        except KeyError:
            self.model_name = 'model'

        try:
            self.mode = options[KEY_MODE]
        except KeyError:
            self.mode = 'train'

        try:
            self.save_dir = options[KEY_SAVE_DIR]
        except KeyError:
            self.save_dir = DATETIME_NOW

        try:
            self.learning_rate = options[KEY_LEARNING_RATE]
        except KeyError:
            self.learning_rate = 0.003

        try:
            self.batch_size = options[KEY_BATCH_SIZE]
        except KeyError:
            self.batch_size = 64

        try:
            self.seq_length = options[KEY_SEQ_LENGTH]
        except KeyError:
            self.seq_length = 5

    def save(self):
        # Save checkpoint.
        self.saver.save(self.session, self.checkpoint_path)
        self.logger.warning("Saver reach checkpoint.")

    def restore(self):
        self.saver.restore(self.session, self.checkpoint_path)


class BaseRLModel(BaseModel):

    def __init__(self, a_space, s_space, **options):
        super(BaseRLModel, self).__init__(**options)
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
            self.train_steps = options[KEY_TRAIN_STEPS]
        except KeyError:
            self.train_steps = 2000

        try:
            self.eval_episodes = options[KEY_EVAL_EPISODE]
        except KeyError:
            self.eval_episodes = 1

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
    def train(self):
        pass

    @abstractmethod
    def predict(self, s):
        pass

    @abstractmethod
    def snapshot(self, s, a, r, s_n):
        pass
