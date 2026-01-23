import matplotlib.pyplot as plt
import core.const as cr
from core.utils import Utils
import numpy as np

class Plotter:
    @staticmethod
    def plot_function(output_plot, output_train):
        
        for i,plot in enumerate(output_plot):
            plt.plot(cr.x_values, plot, label=f"Network output of Neuron {i}")

        plt.scatter(cr.x_train, cr.y_train, color='black', label="Data")

        
        L = Utils.likelihood(output_train, cr.y_train)
        nll = Utils.nll(output_train, cr.y_train)
        plt.title(f"Likelihood on training set: {round(L,5)}\nNegative Log-Likelihood on training set: {round(nll,5)}")

        plt.legend()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
