import random
import math


class Value:
    # noinspection PyProtectedMember
    """
    class to store operands and operation for some expression
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.  # store the derivative of the loss wrt current variable
        self._backward = lambda: None  # find derivative
        self._prev = set(_children)  # store previous objects when operations are called
        self._op = _op  # store operation that created children
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        """keeps track of values and operand that make the output"""
        other = other if isinstance(other, Value) else Value(other)  # value(x) + number(y)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            # need to accumulate in case variable is used twice
            # derivative when adding addition is derivative of output node : d(self)/dy = 1.0
            self.grad += 1. * out.grad
            other.grad += 1. * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad  # derivative when mul is found by chain rule : d(self)/dz*(dz/dy)
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other, modulo=None):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad = other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # x * value
        return self * other  # x + value

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t,
                    _children=(self,),
                    _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad  # grad(tanh) = (1 - tanh**2) * d(out)

        out._backward = _backward
        return out

    def backward(self):
        """
        backpropagation after ordering the graph from children to root
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        topo = []
        visited = set()

        def build_topo(v):
            """
            sort (children -> root)
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    """
    implements a neuron
    """

    def __init__(self, num_inputs):
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inp):
        # w . x +b       call variable(inp) -> output
        activation = sum((wi * xi for wi, xi in zip(self.weight, inp)), self.bias)  # sum(generator, start)
        output = activation.tanh()
        return output

    def parameters(self):
        """
        returns parameters of neuron
        """
        return self.weight + [self.bias]


class Layer:
    """
    single layer of neuron
    """

    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, inp):
        outputs = [n(inp) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        """
        returns parameters of layer
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """
    a multi layer perceptron
    """

    def __init__(self, num_inputs, num_outputs: list):
        sz = [num_inputs] + num_outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(num_outputs))]

    def __call__(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp

    def parameters(self):
        """
        returns parameters of neural network
        """
        return [p for layer in self.layers for p in layer.parameters()]  # list comprehension same as layer.parameters()
