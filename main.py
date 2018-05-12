import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from utility import function
from nn.dense import Dense

np.random.seed(42)

iris = load_iris()

scaler = StandardScaler()
scaler.fit(iris.data)

x_data = scaler.transform(iris.data)
y_data = np.zeros((150, 3))
y_data[np.arange(150), iris.target] = 1

# activation_funcs = [function.tanh] * 1
activation_funcs = [function.relu] * 1
# activation_funcs = [function.sigmoid] * 1
activation_funcs.append(function.linear)

dense = Dense(x_space=4, y_space=3, hidden_units_list=[10], **{
    "loss_func": function.mean_square_error,
    "activation_funcs": activation_funcs,
    "learning_rate": 0.01,
    "enable_logger": True,
    "model_name": 'iris',
    "batch_size": 30,
    'model': 'train',
})

dense.train(x_data, y_data)
# dense.restore()
dense.evaluate(x_data, y_data)
