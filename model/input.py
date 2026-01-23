import numpy as np

class Input:
    def __init__(self, omega, x_values):
        self.omega = omega
        self.x_values = x_values

    def calculate_value(self):
        # Simple linear input: omega * x
        return np.array(self.omega) * np.array(self.x_values)
