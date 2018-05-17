import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from utility import function
from ann.dense import Dense

np.random.seed(135)

data_count = 25

x1_points = np.linspace(0, 10, data_count).reshape((-1, 1))
x2_points = np.multiply(2, x1_points) + np.random.randint(-10, 10, size=(data_count,)).reshape((-1, 1))

x1 = np.concatenate((x1_points, x2_points), axis=1)
y1 = np.array([[1, 0, 0, 0]] * data_count)

x1_points = np.linspace(1, 10, data_count).reshape((-1, 1))
x2_points = np.multiply(-2, x1_points) + np.random.randint(-10, 10, size=(data_count,)).reshape((-1, 1))

x2 = np.concatenate((x1_points, x2_points), axis=1)
y2 = np.array([[0, 1, 0, 0]] * data_count)

x1_points = np.linspace(-1, -10, data_count).reshape((-1, 1))
x2_points = np.multiply(2, x1_points) + np.random.randint(-10, 10, size=(data_count,)).reshape((-1, 1))

x3 = np.concatenate((x1_points, x2_points), axis=1)
y3 = np.array([[0, 0, 1, 0]] * data_count)

x1_points = np.linspace(-1, -10, data_count).reshape((-1, 1))
x2_points = np.multiply(-2, x1_points) + np.random.randint(-10, 10, size=(data_count,)).reshape((-1, 1))

x4 = np.concatenate((x1_points, x2_points), axis=1)
y4 = np.array([[0, 0, 0, 1]] * data_count)

x_data = np.concatenate((x1, x2, x3, x4))
y_data = np.concatenate((y1, y2, y3, y4))

x_train = StandardScaler().fit_transform(x_data)
y_train = y_data

activation_funcs = [function.relu] * 2
# activation_funcs = [function.sigmoid] * 1
activation_funcs.append(function.linear)

dense = Dense(x_space=2, y_space=4, hidden_units_list=[6, 6], **{
    "loss_func": function.softmax_cross_entropy,
    "activation_funcs": activation_funcs,
    "learning_rate": 0.003,
    "enable_logger": True,
    "model_name": 'base',
    "batch_size": 100,
    "max_epoch": 1000,
    'model': 'train',
})

dense.train(x_data, y_data)
# dense.restore()
dense.evaluate(x_data, y_data)

x1_test = np.linspace(-20, 20, 300)
x2_test = np.linspace(-30, 30, 300)

x1_mesh, x2_mesh = np.meshgrid(x1_test, x2_test)

x_test = np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T
y_test = np.argmax(dense.predict(x_test), axis=1)

plt.pcolormesh(x1_mesh, x2_mesh, y_test.reshape(x1_mesh.shape))
plt.scatter(x1[:, 0], x1[:, 1], marker='x')
plt.scatter(x2[:, 0], x2[:, 1], marker='o')
plt.scatter(x3[:, 0], x3[:, 1], marker='*')
plt.scatter(x4[:, 0], x4[:, 1], marker='p')
plt.show()
