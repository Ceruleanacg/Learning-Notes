import matplotlib.pyplot as plt
import numpy as np

data_count = 100

x1_positive = np.linspace(-10, 10, data_count)
x2_positive = 0.3 * x1_positive + 10 + np.random.randint(-5, 5, data_count)
y_positive = np.array([1] * data_count)

x1_negative = np.linspace(-10, 10, data_count)
x2_negative = 0.3 * x1_negative - 10 + np.random.randint(-5, 5, data_count)
y_negative = np.array([-1] * data_count)

x1 = np.concatenate([x1_positive, x1_negative])
x2 = np.concatenate([x2_positive, x2_negative])

y_label = np.concatenate([y_positive, y_negative])

w1 = np.random.normal(0, 0.002)
w2 = np.random.normal(0, 0.002)
b = np.random.normal(0, 0.002)

training_steps = 1000

eta = 0.001

for step in range(training_steps):
    # grad_w1 = np.mean((w1 * x1 + w2 * x2 + b - y_label) * x1)
    # grad_w2 = np.mean((w1 * x1 + w2 * x2 + b - y_label) * x2)
    # grad_b = np.mean(w1 * x1 + w2 * x2 + b)

    hinge_judge_term = y_label * (w1 * x1 + w2 * x2 + b)

    mask_no_grad = hinge_judge_term > 1

    grad_before_mean_w1 = -y_label * x1
    grad_before_mean_w1[mask_no_grad] = 0
    grad_w1 = np.mean(grad_before_mean_w1)

    grad_before_mean_w2 = -y_label * x2
    grad_before_mean_w2[mask_no_grad] = 0
    grad_w2 = np.mean(grad_before_mean_w2)

    grad_before_mean_b = -y_label * 1
    grad_before_mean_b[mask_no_grad] = 0
    grad_b = np.mean(grad_before_mean_b)

    w1 -= eta * grad_w1
    w2 -= eta * grad_w2
    b -= eta * grad_b

plt.scatter(x1_positive, x2_positive, c='r')
plt.scatter(x1_negative, x2_negative, c='b')
plt.plot(x1, -(w1 * x1 + b) / w2, c='g')
plt.show()