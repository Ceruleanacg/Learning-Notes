# coding=utf-8

import multiprocessing as mp
import tensorflow as tf
import logging
import gym

from base.model import *
from playground import PPO
from utility.launcher import start_game


def start_a3c(cluster, role, task_index):
    server = tf.train.Server(cluster, job_name=role, task_index=task_index)
    if role == 'ps':
        logging.warning('Parameter server started.')
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{}".format(task_index),
                                                      cluster=cluster)):

            # Make env.
            env = gym.make('CartPole-v0')
            env.seed(1)
            env = env.unwrapped
            # Init session.
            session = tf.Session(server.target)
            # Init agent.
            agent = PPO.Agent(env.action_space.n, env.observation_space.shape[0], **{
                KEY_SESSION: session,
                KEY_MODEL_NAME: 'PPO',
                KEY_TRAIN_EPISODE: 1000
            })
            start_game(env, agent)


def main():

    cluster = tf.train.ClusterSpec({
        'worker': [
            'localhost:8001',
            'localhost:8002',
            'localhost:8003',
            'localhost:8004',
        ],
        'ps': [
            'localhost:8000'
        ]
    })

    role_task_index_map = [
        ('ps', 0),
        ('worker', 0),
        ('worker', 1),
        ('worker', 2),
        ('worker', 3)
    ]

    pool = mp.Pool(processes=5)

    for role, task_index in role_task_index_map:
        pool.apply_async(start_a3c, args=(cluster, role, task_index, ))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
