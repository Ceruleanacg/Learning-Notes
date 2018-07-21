# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import requests
import logging
import shutil
import os

from static import CHECKPOINTS_DIR

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_save_dir = os.path.join(CHECKPOINTS_DIR, 'TensorFlowServing')
graph_save_dir = os.path.join(CHECKPOINTS_DIR, 'TensorFlowServing', 'graph')

# x_test = np.linspace(2 * -np.pi, 2 * np.pi, num=100).reshape((-1, 1))
# y_test = np.sin(x_test)

# x_train = x_test + np.random.normal(0.3, 0.003)
# y_train = np.sin(x_train) + np.random.normal(0.0, 0.00003)

# x_train = x_train.astype(np.float32)
# y_train = y_train.astype(np.float32)

x_train = np.load(os.path.join(data_save_dir, 'x_train.npy')).astype(np.float32)
y_train = np.load(os.path.join(data_save_dir, 'y_train.npy')).astype(np.float32)

np.save(os.path.join(data_save_dir, 'x_train.npy'), x_train)
np.save(os.path.join(data_save_dir, 'y_train.npy'), y_train)


def train():

    session = tf.Session()

    x_input = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_input = tf.placeholder(tf.float32, [None, 1], name='y_input')

    fc1 = tf.layers.dense(x_input, 10, tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 10, tf.nn.relu)

    y_predict = tf.layers.dense(fc2, 1)

    loss_func = tf.losses.mean_squared_error(labels=y_input, predictions=y_predict)

    optimizer = tf.train.AdamOptimizer().minimize(loss_func)

    session.run(tf.global_variables_initializer())

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'x_input': tf.saved_model.utils.build_tensor_info(x_input),
            'y_input': tf.saved_model.utils.build_tensor_info(y_input)
        },
        outputs={
            'y_predict': tf.saved_model.utils.build_tensor_info(y_predict),
            'loss_func': tf.saved_model.utils.build_tensor_info(loss_func)
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    for step in range(2000):
        session.run(optimizer, {
            x_input: x_train,
            y_input: y_train
        })
        if (step + 1) % 500 == 0:
            if os.path.exists(graph_save_dir):
                shutil.rmtree(graph_save_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(graph_save_dir)
            builder.add_meta_graph_and_variables(session,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
            # builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], {'signature': signature})
            builder.save()

    loss = session.run(loss_func, {
        x_input: x_train,
        y_input: y_train
    })

    logging.warning('Loss: {}'.format(loss))
    # builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.TRAINING], {'signature': signature})
    # builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], {'signature': signature})
    # builder.save()


def test():
    # Session.
    session = tf.Session()
    # Load meta graph.
    meta_graph_def = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], graph_save_dir)  # type: tf.MetaGraphDef
    # Get signature.
    signature_def = meta_graph_def.signature_def
    signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # Get input tensor.
    x_input_tensor = signature.inputs['x_input'].name
    y_input_tensor = signature.inputs['y_input'].name
    # Get output tensor.
    y_predict_tensor = signature.outputs['y_predict'].name
    # Get loss func.
    loss_op = signature.outputs['loss_func'].name

    _, loss = session.run([y_predict_tensor, loss_op], {
        x_input_tensor: x_train,
        y_input_tensor: y_train,
    })

    logging.warning('Loss: {}'.format(loss))


def inference_v1():
    # Init channel.
    channel = implementations.insecure_channel('localhost', 9000)
    # Init stub.
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Init request.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['x_input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_train, shape=x_train.shape)
    )
    request.inputs['y_input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(y_train, shape=y_train.shape)
    )
    # Predict.
    future = stub.Predict.future(request, 2.0)
    result = future.result().outputs['loss_func'].float_val
    logging.warning('Loss: {}'.format(result))


def inference_v2():
    # Init url.
    url = "http://localhost:9001/v1/models/test:predict"
    # url = 'http://172.16.11.43:10000/tool_list/test'
    # Init body.
    import json
    body = {
        # 'signature_name': tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        'instances':  [
            {
                'x_input': json.dumps(x_train.tolist(), ensure_ascii=True)
            }
        ]
    }
    # Post.
    # response = requests.post(url, data=body)
    response = requests.post(url, json=json.dumps(body))
    logging.warning('{}'.format(response.text))
    return response


train()
# test()
# inference_v1()
# inference_v2()


# plt.plot(y_train)
# plt.plot(y_test)
# plt.show()

