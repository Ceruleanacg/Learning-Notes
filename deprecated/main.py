import tensorflow as tf

from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()

x_data = tf.random_normal([1000, ])
x_noise = tf.random_normal([1000, ])

y_label = 3 * x_data + x_noise

w = tfe.Variable(5.)
b = tfe.Variable(10.)


def mse(label, predict):
    loss = tf.losses.mean_squared_error(label, predict)
    return loss


optimizer = tf.train.GradientDescentOptimizer(0.003)

for step in range(3000):
    with tf.GradientTape(persistent=True) as tape:
        y_predict = w * x_data + b
        l = mse(y_label, y_predict)
        w_grad, b_grad = tape.gradient(l, [w, b])
    optimizer.apply_gradients(zip([w_grad, b_grad], [w, b]))
    if step % 100 == 0:
        print(l)
