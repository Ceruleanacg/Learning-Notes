import numpy as np
import json
import os

from static import CKPT_DIR
from utility import function
from utility.logger import generate_model_logger


class Dense(object):

    def __init__(self, x_space, y_space, hidden_units_list, **options):

        # Init x space, y space.
        self.x_space = x_space
        self.y_space = y_space

        # Init layer & neuron info.
        self.hidden_units_list = hidden_units_list
        self.hidden_layer_count = len(hidden_units_list)
        self.total_layer_count = self.hidden_layer_count + 1

        # Init weights, biases.
        self.weights, self.biases = {}, {}

        # Init a, z, outputs caches.
        self.z_outputs, self.z_inputs = {}, {}

        # Init deltas caches.
        self.deltas = {}

        self._validate_parameters()
        self._init_func_map()
        self._init_options(options)
        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        # Hidden Layer.
        for index, hidden_units in enumerate(self.hidden_units_list):
            # x_space is the shape of last layer, and the shape of weight of current layer.
            x_space = self.x_space if index == 0 else self.hidden_units_list[index - 1]
            # hidden_units is shape of current layer, also neuron count.
            weights, biases = np.random.normal(0, 0.01, (hidden_units, x_space)), np.zeros((hidden_units, 1))
            self.weights[index], self.biases[index] = weights, biases
        # Output Layer.
        x_space = self.hidden_units_list[-1]
        weights, biases = np.random.normal(0, 0.01, (self.y_space, x_space)), np.zeros((self.y_space, 1))
        self.weights[self.total_layer_count - 1], self.biases[self.total_layer_count - 1] = weights, biases

    def _validate_parameters(self):
        if self.hidden_layer_count == 0 or len(self.hidden_units_list) == 0:
            raise ValueError('Layer count or neuron count list cannot be zero.')
        if self.hidden_layer_count != len(self.hidden_units_list):
            raise ValueError('Layer count should be equal to length of neuron count list.')

    def _init_func_map(self):
        # Init Activation Func and Grad Map.
        self.activation_grad_map = {
            function.relu: np.vectorize(function.grad_relu),
            function.tanh: np.vectorize(function.grad_tanh),
            function.linear: np.vectorize(function.grad_linear),
            function.sigmoid: np.vectorize(function.grad_sigmoid),
        }
        self.grad_loss_map = {
            function.softmax_cross_entropy: function.grad_softmax_cross_entropy,
            function.mean_square_error: function.grad_mean_square_error
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

        try:
            self.loss_func = options['loss_func']
        except KeyError:
            self.loss_func = function.mean_square_error
        finally:
            self.grad_func = self.grad_loss_map[self.loss_func]
            # Enable softmax.
            if self.grad_func == self.grad_loss_map[function.softmax_cross_entropy]:
                self.enable_softmax = True
            else:
                self.enable_softmax = False

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
        z_input = input_batch
        # Forward layer by layer.
        for layer_index in range(self.total_layer_count):
            # Get weights and biases.
            weights, biases = self.weights[layer_index], self.biases[layer_index]
            # Save result as grad w.
            self.z_inputs[layer_index] = z_input
            z_output = np.dot(z_input, weights.T) + biases.T
            # Save result of a for backward.
            self.z_outputs[layer_index] = z_output
            # z_input is also called a_output.
            z_input = self.activation_funcs[layer_index](z_output)
        return z_input

    def _backward(self, error):
        # error here is shape of (batch_size, y_space)
        for index in np.arange(0, self.total_layer_count)[::-1]:
            # dl/dw = dz/dw * da/dz * (dl/da) | x = x_batch.
            z_outputs = self.z_outputs[index]
            # Get grad of activation func.
            grad_activation_func = self.grad_activation_funcs[index]
            # Calculate da/dz.
            grad_z_batch = grad_activation_func(z_outputs)
            # Calculate dl/da * da/dz.
            delta = error * grad_z_batch
            # Save delta.
            self.deltas[index] = delta
            # Update error, dz/da
            error = np.dot(delta, self.weights[index])

    def _update_weights_and_biases(self):
        for index in range(self.total_layer_count):
            # Get z_input and delta.
            z_input, delta = self.z_inputs[index], self.deltas[index]
            # Calculate grad weights, grad biases, dl/da * da/dz * dz/dw
            grad_weights = -np.dot(delta.T, z_input)
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
                # Calculate loss.
                loss = self.loss_func(y_predict, y_batch)
                epoch_loss.append(loss)
                # Calculate error.
                error = self.grad_func(y_predict, y_batch)
                # Bp & Update.
                self._backward(error)
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
        if self.enable_softmax:
            result = function.softmax(self._forward(x_batch))
        else:
            result = self._forward(x_batch)
        return result

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
