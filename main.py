import numpy as np
import core.const as cr
import matplotlib.pyplot as plt
from model.input import Input
from model.perceptron import Perceptron
from core.utils import Utils
from core.plotter import Plotter

p1_train = Perceptron([Input(cr.omega_0[0], cr.x_train)], cr.beta_0[0], Utils.relu)
p2_train = Perceptron([Input(cr.omega_0[1], cr.x_train)], cr.beta_0[1], Utils.relu)
p3_train = Perceptron([Input(cr.omega_0[2], cr.x_train)], cr.beta_0[2], Utils.relu)

o1_train = np.array(p1_train.output)
o2_train = np.array(p2_train.output)
o3_train = np.array(p3_train.output)

p4_train = Perceptron([o1_train, o2_train, o3_train], cr.beta_1, Utils.sigmoid, weights=cr.omega_1)
p4_output_train = np.array(p4_train.output)  

p1_plot = Perceptron([Input(cr.omega_0[0], cr.x_values)], cr.beta_0[0], Utils.relu)
p2_plot = Perceptron([Input(cr.omega_0[1], cr.x_values)], cr.beta_0[1], Utils.relu)
p3_plot = Perceptron([Input(cr.omega_0[2], cr.x_values)], cr.beta_0[2], Utils.relu)

o1_plot = np.array(p1_plot.output)
o2_plot = np.array(p2_plot.output)
o3_plot = np.array(p3_plot.output)

p4_plot = Perceptron([o1_plot, o2_plot, o3_plot], cr.beta_1, Utils.sigmoid, weights=cr.omega_1)
p4_output_plot = np.array(p4_plot.output)  

Plotter.plot_function([p1_plot.output,p2_plot.output,p3_plot.output,p4_output_plot], p4_output_train)

beta1_variance = np.linspace(0.01, 20, 1000)
log_likelihood_array = np.array([])
nll_array = np.array([])

for i in beta1_variance:
    p4_with_beta1_variance = Perceptron([o1_train, o2_train, o3_train], i, Utils.sigmoid, weights=cr.omega_1)
    log_likelihood_array = np.append(log_likelihood_array,Utils.likelihood(p4_with_beta1_variance.output,cr.y_train))
    nll_array = np.append(nll_array,Utils.nll(p4_with_beta1_variance.output,cr.y_train))
i1_max = np.argmax(log_likelihood_array)
i2_min = np.argmin(nll_array)

likelihood_array_max = log_likelihood_array[i1_max]
x_at_likelihood_array_max = beta1_variance[i1_max]

nll_array_min = nll_array[i2_min]
x_at_nll_array_min = beta1_variance[i2_min]

plt.plot(beta1_variance, log_likelihood_array, label="Likelihood")
plt.plot(beta1_variance, nll_array, label="NLL")
plt.title(f"Beta1-NLL-L relationship\nMax at LogLikelihood = ({x_at_likelihood_array_max},{likelihood_array_max}) \n Min at Negative LogLikelihood = ({x_at_nll_array_min},{nll_array_min})")

plt.legend()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()



