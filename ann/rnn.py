import numpy as np

from utility import function


class RNN(object):

    def __init__(self, hidden_size, seq_length, x_space, y_space, **options):

        self.x_space = x_space
        self.y_space = y_space

        self.seq_length = seq_length

        self.hidden_size = hidden_size

        self.x_weights = np.zeros((hidden_size, x_space))
        self.s_weights = np.zeros((hidden_size, y_space))
        self.u_weights = np.zeros((y_space, hidden_size))

        self.z_inputs = {}
        self.h_inputs = {}
        self.p_outputs = {}
        self.deltas = {}

        self._init_options(options)
        self._init_weights_and_biases()

    def _init_options(self, options):

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 16
        finally:
            if self.batch_size < 1:
                raise ValueError('Batch size must larger than 1.')

    def _init_weights_and_biases(self):
        self.x_weights[:, ] = np.random.normal(loc=0.0, scale=0.001)
        self.s_weights[:, ] = np.random.normal(loc=0.0, scale=0.001)
        self.u_weights[:, ] = np.random.normal(loc=0.0, scale=0.001)

    def _forward(self, input_batch):
        # Initialize s_t
        s_t = np.zeros((input_batch.shape[0], self.y_space))
        # Forward pass.
        for seq_index in range(self.seq_length):
            # Get x_t.
            x_t = input_batch[:, seq_index, :]
            # Save dz/dw
            self.z_inputs[seq_index] = x_t
            # (batch_size, x_space) * (x_space, hidden_size) -> (batch_size, hidden_size)
            z_t = np.dot(x_t, self.x_weights.T)
            # Save dh/ds
            self.h_inputs[seq_index] = s_t
            # (batch_size, y_space) * (y_space, hidden_size) -> (batch_size, hidden_size)
            h_t = np.dot(s_t, self.s_weights.T)
            # (batch_size, hidden_size) * (hidden_size, y_space) -> (batch_size, y_space)
            phi_t = np.dot((z_t + h_t), self.u_weights.T)
            # Save da/dp
            self.p_outputs = phi_t
            # Get s_t
            s_t = function.tanh(phi_t)
        return s_t

    def _backward(self, error):
        for seq_index in range(self.seq_length)[::-1]:
            z_input = self.z_inputs[seq_index]
            h_input = self.h_inputs[seq_index]
            # da/dp
            p_output = self.p_outputs[seq_index]
            # dp/dz
            p_input = self.p_inputs[seq_index]
            # TODO - Implements

    def train(self, x_data, y_data):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
