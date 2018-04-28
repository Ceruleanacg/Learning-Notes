import numpy as np
import json
import os

from static import CKPT_DIR
from utility import function
from utility.logger import generate_model_logger


class Dense(object):

    def __init__(self, x_space, y_space, neuron_count_list, **options):

        # Init x space, y space.
        self.x_space = x_space
        self.y_space = y_space

        # Init layer & neuron info.
        self.neuron_count_list = neuron_count_list
        self.hidden_layer_count = len(neuron_count_list)
        self.total_layer_count = self.hidden_layer_count + 1

        # Init weights, biases.
        self.weights, self.biases = {}, {}

        # Init a, z, outputs caches.
        self.a_outputs, self.z_outputs = {}, {}

        # Init deltas caches.
        self.deltas = {}

        self._validate_parameters()
        self._init_func_map()
        self._init_options(options)
        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        # Hidden Layer.
        for index, neuron_count in enumerate(self.neuron_count_list):
            x_space = self.x_space if index == 0 else self.neuron_count_list[index - 1]
            weights, biases = np.random.normal(0, 0.01, (neuron_count, x_space)), np.zeros((neuron_count, 1))
            self.weights[index], self.biases[index] = weights, biases
        # Output Layer.
        x_space = self.neuron_count_list[-1]
        weights, biases = np.random.normal(0, 0.01, (self.y_space, x_space)), np.zeros((self.y_space, 1))
        self.weights[self.total_layer_count - 1], self.biases[self.total_layer_count - 1] = weights, biases

    def _validate_parameters(self):
        if self.hidden_layer_count == 0 or len(self.neuron_count_list) == 0:
            raise ValueError('Layer count or neuron count list cannot be zero.')
        if self.hidden_layer_count != len(self.neuron_count_list):
            raise ValueError('Layer count should be equal to length of neuron count list.')

    def _init_func_map(self):
        # Init Activation Func and Grad Map.
        self.activation_grad_map = {
            function.relu: np.vectorize(function.grad_relu),
            function.tanh: np.vectorize(function.grad_tanh),
            function.linear: np.vectorize(function.grad_linear),
            function.sigmoid: np.vectorize(function.grad_sigmoid),
        }

    def _init_options(self, options):

        try:
            self.model_name = options['model_name']
        except KeyError:
            self.model_name = 'model'
        finally:
            if not isinstance(self.model_name, str):
                raise ValueError('Model name must be a str.')

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

        # Init Activation Func and Grad Func.
        try:
            self.activation_funcs = options['activation_funcs']
        except KeyError:
            self.activation_funcs = [function.tanh] * self.hidden_layer_count
            self.activation_funcs.append(function.linear)
        finally:
            if len(self.activation_funcs) != self.total_layer_count:
                raise ValueError('Activation func count should be equal to total layer count.')

        try:
            self.grad_activation_funcs = [self.activation_grad_map[act_func] for act_func in self.activation_funcs]
            self.activation_funcs = [np.vectorize(act_func) for act_func in self.activation_funcs]
        except KeyError:
            raise KeyError('Grad func not exists.')

        # Init Batch Size.
        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 16
        finally:
            if self.batch_size < 1:
                raise ValueError('Batch size must larger than 1.')

        # Init Learning Rate.
        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.003
        finally:
            if self.learning_rate < 0.0:
                raise ValueError('Learning rate must be positive.')

        try:
            self.loss_func = options['loss_func']
        except KeyError:
            self.loss_func = function.mean_square_error

        try:
            self.max_epoch = options['max_epoch']
        except KeyError:
            self.max_epoch = 3000
        finally:
            if self.max_epoch < 1:
                raise ValueError('Epoch must be larger than 1.')

        try:
            self.enable_logger = options['enable_logger']
        except KeyError:
            self.enable_logger = True
        finally:
            if self.enable_logger:
                self.logger = generate_model_logger(self.model_name)

        self.history_loss = []

    def _forward(self, input_batch):
        # Temporal result, a_batch.
        z_output = input_batch
        # Forward layer by layer.
        for index in range(self.total_layer_count):
            # Get weights and biases.
            weights, biases = self.weights[index], self.biases[index]
            # Save result as grad.
            self.z_outputs[index] = z_output
            a_output = np.dot(z_output, weights.T) + biases.T
            self.a_outputs[index] = a_output
            z_output = self.activation_funcs[index](a_output)
        return z_output

    def _backward(self, diff):
        # Backward error.
        error = diff
        for index in np.arange(0, self.total_layer_count)[::-1]:
            # dl/dw = da/dw * dz/da * (dl/dz) | x = x_batch.
            a_batch = self.a_outputs[index]
            # Calculate dz/da.
            grad_a_batch = self.grad_activation_funcs[index](a_batch)
            # Calculate dy/dz.
            delta = np.multiply(error, grad_a_batch)
            # Save delta.
            self.deltas[index] = delta
            # Backward error.
            error = np.dot(delta, self.weights[index])

    def _update_weights_and_biases(self):
        for index in range(self.total_layer_count):
            # Get z_output and delta.
            z_output, delta = self.z_outputs[index], self.deltas[index]
            # Calculate grad weights, grad biases.
            grad_weights = -np.dot(delta.T, z_output)
            grad_biases = -np.mean(delta, axis=0).reshape(self.biases[index].shape)
            # Update weights, biases.
            self.weights[index] -= self.learning_rate * grad_weights
            self.biases[index] -= self.learning_rate * grad_biases

    def train(self, x_data, y_data):
        iteration, epoch, x_data_count = 0, 0, len(x_data)
        while epoch < self.max_epoch:
            s_index, e_index, epoch_loss = 0, self.batch_size, []
            while True:
                # Generate batch x, y
                x_batch, y_batch = x_data[s_index: e_index], y_data[s_index: e_index]
                # Calculate y_predict.
                y_predict = self._forward(x_batch)
                # Calculate diff.
                diff = y_batch - y_predict
                # Calculate loss.
                loss = self.loss_func(y_predict, y_batch)
                epoch_loss.append(loss)
                # Bp & Update.
                self._backward(diff)
                self._update_weights_and_biases()
                # Update index.
                s_index += self.batch_size
                e_index = s_index + self.batch_size
                # Add iteration.
                iteration += 1
                if e_index > len(x_data):
                    mean_epoch_loss = np.mean(epoch_loss)
                    self.history_loss.append(mean_epoch_loss)
                    break
            if epoch % 100 == 0:
                self.save()
                self.evaluate(x_data, y_data)
                self.logger.warning("Epoch: {:d} | loss: {:.6f}".format(epoch, mean_epoch_loss))
            epoch += 1

    def predict(self, x_batch):
        return self._forward(x_batch)

    def evaluate(self, x_data, y_data):
        y_label, y_output = np.argmax(y_data, axis=1), np.argmax(self.predict(x_data), axis=1)
        self.logger.warning("Accuracy: {:.3f} ".format(np.sum(y_label == y_output) / len(x_data)))

    def save(self):
        save_dir = os.path.join(CKPT_DIR, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'weights.json'), 'w') as fp:
            weights = [weights.tolist() for weights in self.weights.values()]
            json.dump(weights, fp, indent=True)
        with open(os.path.join(save_dir, 'biases.json'), 'w') as fp:
            biases = [biases.tolist() for biases in self.biases.values()]
            json.dump(biases, fp, indent=True)
        self.logger.warning("Model saved.")

    def restore(self):
        save_dir = os.path.join(CKPT_DIR, self.model_name)
        try:
            with open(os.path.join(save_dir, 'weights.json'), 'r') as fp:
                weights = json.load(fp)
                for index in range(self.total_layer_count):
                    self.weights[index] = np.array(weights[index])
        except FileNotFoundError:
            raise FileNotFoundError('Weights not exists.')

        try:
            with open(os.path.join(save_dir, 'biases.json'), 'r') as fp:
                biases = json.load(fp)
                for index in range(self.total_layer_count):
                    self.biases[index] = np.array(biases[index])
        except FileNotFoundError:
            raise FileNotFoundError('biases not exists.')

        self.logger.warning("Model restored.")
