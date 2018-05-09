from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from utility import function

data_count = 100

x1_points = np.linspace(-5, 5, data_count).reshape((-1, 1))
x2_points = np.multiply(5, x1_points) + np.random.randint(-5, 50, size=(data_count,)).reshape((-1, 1))

x_positive_data = np.concatenate((x1_points, x2_points), axis=1)
y_positive_data = np.array([[1, 0]] * data_count)

x1_points = np.linspace(-5, 5, data_count).reshape((-1, 1))
x2_points = np.multiply(5, x1_points) - np.random.randint(-5, 50, size=(data_count,)).reshape((-1, 1))

x_negative_data = np.concatenate((x1_points, x2_points), axis=1)
y_negative_data = np.array([[0, 1]] * data_count)

x_data = np.concatenate((x_positive_data, x_negative_data))
y_data = np.concatenate((y_positive_data, y_negative_data))

sigmoid = np.vectorize(function.sigmoid)
grad_sigmoid = np.vectorize(function.grad_sigmoid)

w = np.array([5, 5])

y = - np.multiply(x_data[: data_count, 0], w[0]) / w[1]

plt.figure(figsize=(16, 9))
plt.plot(x_data[: data_count, 0], y)
plt.scatter(x_data[:data_count, 0], x_data[:data_count, 1], s=50, color='g', marker='o')
plt.scatter(x_data[data_count:, 0], x_data[data_count:, 1], s=50, color='r', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

loss_cache = []

for iteration in range(10000):
    # 计算内积 (200, 2) * (2, 1) -> (200, 1)
    y_product = np.dot(x_data, w.T).reshape((-1, 1))
    # 计算预测标签值
    y_positive_predict = sigmoid(y_product).reshape((-1, 1))
    y_negative_predict = 1 - y_positive_predict
    y_predict = np.concatenate((y_positive_predict, y_negative_predict), axis=1)
    # y_predict[y_predict < 1e-10] = 1e-10
    # 计算交叉熵
    negative_log = -np.log(y_predict)
    cross_entropy = np.mean(np.sum(y_data * negative_log))
    # 计算梯度 (200 ,2) / (200 ,2) * (200, 1) * (2, 200) ->
    partial_c = y_data / y_predict
    partial_s = grad_sigmoid(y_product)
    partial_h = x_data
    partial_product = partial_c * partial_s * partial_h
    grad_w = -np.mean(partial_product.reshape((2, -1)), axis=1)
    # 更新梯度
    w = w - 0.3 * grad_w
    # 缓存交叉熵
    loss_cache.append(cross_entropy)

y = - np.multiply(x_data[: data_count, 0], w[0]) / w[1]
#
# w1_sample = np.linspace(-10, 10, 2 * data_count).reshape((-1, 1))
# w2_sample = np.linspace(-10, 10, 2 * data_count).reshape((-1, 1))
#
# w_sample = np.concatenate((w1_sample, w2_sample), axis=1)
#
# # (200, 2) * (2, 200) -> (200 * 200)
# loss = y_data * np.log(sigmoid(np.dot(x_data, w_sample.T)))
#
# figure = plt.figure(figsize=(16, 6))
# axes = Axes3D(figure)
# axes.set_xlabel('w')
# axes.set_ylabel('b')
# axes.plot_surface(w1_sample.T, w2_sample, loss, cmap='rainbow')

plt.figure(figsize=(16, 9))
plt.title('CrossEntropy')
plt.plot(loss_cache)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(x_data[: data_count, 0], y)
plt.scatter(x_data[:data_count, 0], x_data[:data_count, 1], s=50, color='g', marker='o')
plt.scatter(x_data[data_count:, 0], x_data[data_count:, 1], s=50, color='r', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
