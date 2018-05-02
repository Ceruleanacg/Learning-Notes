from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data_count = 100

x_data = np.linspace(-20, 20, data_count)
y_data = np.multiply(2, x_data) + 3 + np.random.normal(loc=0, scale=1.0, size=(data_count,))

x_data = x_data.reshape((-1, 1))
y_data = y_data.reshape((-1, 1))

# w = np.random.normal(size=(1, 1))
# b = np.random.normal(size=(1, 1))
w = 10
b = 20
y_predict = np.dot(x_data, w) + b

w_sample = np.linspace(-10, 10, data_count).reshape((-1, 1))
b_sample = np.linspace(-10, 10, data_count).reshape((-1, 1))

x_data = x_data.reshape((-1, 1))
y_data = y_data.reshape((-1, 1))

loss = np.square(np.dot(w_sample, x_data.T) + b_sample - y_data) / data_count

w_cache, b_cache, l_cache, = [], [], []

for iteration in range(2000):
    y_predict = w * x_data + b
    diff = y_predict - y_data
    grad_w = np.mean(diff * x_data)
    grad_b = np.mean(diff)
    w -= 0.003 * grad_w
    b -= 0.003 * grad_b
    w_cache.append(w)
    b_cache.append(b)
    l_cache.append(np.mean(diff))

w_cache = np.array(w_cache).reshape((-1,))
b_cache = np.array(w_cache).reshape((-1,))
l_cache = np.array(w_cache).reshape((-1,))


figure = plt.figure(figsize=(16, 9))
figure = Axes3D(figure)
figure.set_xlabel('w')
figure.set_ylabel('b')
figure.plot_surface(w_sample.T, b_sample, loss, cmap='rainbow')
figure.scatter3D(w_cache, b_cache, l_cache, cmap='rainbow')

y_predict = w * x_data + b

plt.figure(figsize=(16, 9))
plt.scatter(x_data, y_data, s=10, color='g')
plt.plot(x_data, y_predict)
plt.title('y=2x+3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
