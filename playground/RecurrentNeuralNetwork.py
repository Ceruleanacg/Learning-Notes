import numpy as np

from ann.rnn import RNN

seq_length = 3

rnn = RNN(hidden_size=2, seq_length=3, x_space=1, y_space=1)

data_count = 100

x_points = np.linspace(-np.pi, np.pi, data_count)

x_data = []
y_data = []

for index in range(data_count):
    if index < seq_length:
        continue
    x_data.append(x_points[index - seq_length: index])
    y_data.append(np.sin(x_points[index - 1]))

x_data = np.array(x_data).reshape((-1, seq_length, 1))
y_data = np.array(y_data).reshape((-1, 1))

print(rnn._forward(x_data))