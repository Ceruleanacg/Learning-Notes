import numpy as np


class Node(object):
    def __init__(self, i, j, c1, c2, l_node=None, r_node=None):
        self.i = i
        self.j = j
        self.c1 = c1
        self.c2 = c2
        self.offset = 0
        self.l_node = l_node
        self.r_node = r_node


class RegressionTree(object):

    def __init__(self):
        self._tree = None
        self.x_data = None
        self.y_data = None

    def fit(self, x, y, max_depth=3):
        self.x_data = x
        self.y_data = y
        # Calculate nodes.
        num_nodes = 2 ** max_depth - 1
        # Init root node.
        root_node = self.make_node(x, y)

        def _fit(_x, _y, _node, _num_nodes):

            if _num_nodes <= 0:
                return

            # Make R.
            x_r1, y_r1 = _x[:_node.i], _y[:_node.i]
            x_r2, y_r2 = _x[_node.i:], _y[_node.i:]

            # Make left node.
            l_node = self.make_node(x_r1, y_r1)

            _node.l_node = l_node

            if _node.l_node:
                # Update offset.
                # l_node.offset = _node.offset

                _num_nodes -= 1
                _fit(x_r1, y_r1, _node.l_node, _num_nodes)

            # Make right node.
            r_node = self.make_node(x_r2, y_r2)

            _node.r_node = r_node

            if _node.r_node:
                # Update offset.
                r_node.offset = _node.i + _node.offset

                _num_nodes -= 1
                _fit(x_r2, y_r2, _node.r_node, _num_nodes)

        _fit(x, y, root_node, num_nodes)

        self._tree = root_node

    def predict(self, x):

        node = self._tree

        def _predict(_x, _node):

            val = self.x_data[_node.i + _node.offset, _node.j]

            if _x[_node.j] < val:
                if _node.l_node:
                    return _predict(_x, _node.l_node)
                else:
                    return _node.c1
            else:
                if _node.r_node:
                    return _predict(_x, _node.r_node)
                else:
                    return _node.c2

        return _predict(x, node)

    @staticmethod
    def make_node(x, y):
        # Get shape.
        rows, cols = x.shape
        if rows <= 1:
            return None
        # Init params.
        best_i, best_j = 1, 1
        best_c1, best_c2 = 0, 0
        best_loss = np.inf
        # Find best split.
        for i in range(1, rows):
            for j in range(0, cols):
                # Calculate c1, c2, loss.
                c1 = np.mean(y[:i])
                c2 = np.mean(y[i:])
                loss = np.mean(y[:i] - c1) + np.mean(y[i:] - c2)
                # Update best if need.
                if loss < best_loss:
                    best_loss = loss
                    best_i = i
                    best_j = j
                    best_c1 = c1
                best_c2 = c2
        node = Node(best_i, best_j, best_c1, best_c2)
        return node


data_x = np.linspace(-10, 10, 20).reshape((-1, 1))
# data_y = np.linspace(-20, 20, 100) + np.random.normal(loc=0, scale=3.5, size=(100, ))
data_y = 2 * data_x

# data_x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape((-1, 1))
# data_y = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])


t = RegressionTree()
t.fit(data_x, data_y, max_depth=8)
# print(t.predict([-4]))
# print(t.predict([1]))
# print(t.predict([2]))
# print(t.predict([3]))
# print(t.predict([4]))
# print(t.predict([20]))
print(t.predict([4.]))
# print([t.predict(data_x[i, :].reshape((-1, ))) for i in range(0, 100)])
