import numpy as np

class Perceptron:
    def __init__(self, inputs, bias, func, weights=None):
        self.inputs = inputs
        self.bias = bias
        self.func = func
        self.weights = weights if weights is not None else [1.0] * len(inputs)
        self.calculate_output()

    def calculate_output(self):
        arrays = []
        for i in self.inputs:
            if hasattr(i, "calculate_value"):
                arrays.append(np.array(i.calculate_value()))
            else:
                arrays.append(np.array(i))

        total_input = np.zeros_like(arrays[0])
        for w, arr in zip(self.weights, arrays):
            total_input += w * arr
        total_input += self.bias

        self.output = self.func(total_input).tolist()
