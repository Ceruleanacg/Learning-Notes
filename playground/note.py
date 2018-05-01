import matplotlib.pyplot as plt
import numpy as np

data_count = 100

x_data = np.linspace(-20, 20, data_count)
y_data = np.multiply(2, x_data) + 3 + np.random.normal(loc=0, scale=1.0, size=(data_count,))

x_data = x_data.reshape((-1, 1))
y_data = y_data.reshape((-1, 1))

w = np.random.normal(size=(1, 1))
b = np.random.normal(size=(1, 1))
y_predict = np.dot(x_data, w) + b


for iteration in range(3000):
    y_predict = w * x_data + b
    grad_w = np.mean((y_predict - y_data) * x_data)
    grad_b = np.mean((y_predict - y_data))
    w -= 0.003 * grad_w
    b -= 0.003 * grad_b

y_predict = w * x_data + b

plt.figure(figsize=(16, 9))
plt.scatter(x_data, y_data, s=10, color='g')
plt.plot(x_data, y_predict)
plt.title('y=2x+3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()