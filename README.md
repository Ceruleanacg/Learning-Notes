# Descent - Learning Notes of DRL & DL

Descent is a repo of Learning notes of DRL & DL, theory, codes, models and notes maybe.

# Content  

## Notes

- [LinearRegression](/note/LinearRegression.ipynb)
- [LogisticRegression](/note/LogisticRegression.ipynb)
- [NeuralNetwork](/note/NeuralNetwork.ipynb)

## Codes

- [Artifical Neuron Network (ANN)](/ann/dense.py)   

# Requirements
- numpy
- scipy
- sklearn
- matplotlib

# Instructions for codes

### [Artifical Neuron Network (ANN)](/ann/dense.py) 

1. Load your data, for example, iris data set.
```
from sklearn.datasets import load_iris
iris = load_iris()
```
2. Standardize your data.
```
scaler = StandardScaler()
scaler.fit(iris.data)

x_data = scaler.transform(iris.data)
y_data = np.zeros((150, 3))
y_data[np.arange(150), iris.target] = 1
``` 
3. Initialize activations, which are configurable.
```
activation_funcs = [function.relu] * 1
# activation_funcs = [function.tanh] * 1
# activation_funcs = [function.sigmoid] * 1
activation_funcs.append(function.linear)
```
4. Initialize model, option parameters are configurable.
```
dense = Dense(x_space=4, y_space=3, neuron_count_list=[10], **{
    "loss_func": function.softmax_cross_entropy,
    "activation_funcs": activation_funcs,
    "learning_rate": 0.01,
    "enable_logger": True,
    "model_name": 'iris',
    "batch_size": 30,
    'model': 'train'
)
```
5. Train or Restore & Evaluate.
```
dense.train(x_data, y_data)
# dense.restore()
dense.evaluate(x_data, y_data)
```