import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

# 定义序列长度
seq_length = 5

# 生成数据点
x_points = np.linspace(-np.pi, np.pi, 200)

x_data = []
y_data = []

# 生成训练数据与标签，训练数据是序列
for index, x in enumerate(x_points):
    # 跳过前5个，因为序列长度是5
    if index < seq_length:
        continue
    # 生成序列
    x_data.append(x_points[index - seq_length: index])
    y_data.append(np.sin(x_points[index - 1]))

# 标准化维度
x_data = np.array(x_data).reshape((-1, 5, 1))
y_data = np.array(y_data).reshape((-1, 1))

# 获取Session
session = tf.Session()

# 定义输入输出张量
x_train = tf.placeholder(tf.float32, [None, 5, 1])
y_train = tf.placeholder(tf.float32, [None, 1])

# 定义LSTM-Cell
cell = rnn.LSTMCell(num_units=4, activation=tf.tanh)

# 获取LSTM输出与状态
cell_output, _ = tf.nn.dynamic_rnn(cell, x_train, dtype=tf.float32)
cell_output = cell_output[:, -1]

# 定义全连接
y_predict = tf.layers.dense(cell_output, 1)

# 定义损失函数
loss_func = tf.losses.mean_squared_error(y_train, y_predict)

# 定义优化器
optimizer = tf.train.AdamOptimizer().minimize(loss_func)

# 初始化张量变量
session.run(tf.global_variables_initializer())

# 训练
for iteration in range(3000):
    _, loss = session.run([optimizer, loss_func], {x_train: x_data, y_train: y_data})
    if iteration % 100 == 0:
        print('Iteration: {} | Loss: {}'.format(iteration, loss))

# 获取输出
y_result = session.run(y_predict, {x_train: x_data})

# 绘图
plt.figure(figsize=(16, 9))
plt.scatter(x_points, np.sin(x_points), s=50, color='g', marker='o')
plt.scatter(x_points[seq_length:], y_result, s=50, color='r', marker='8')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
