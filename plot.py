import numpy as np
import matplotlib
import matplotlib.pyplot as plt


y_pred = np.loadtxt("./y_pred.csv", delimiter=",")
y_real = np.loadtxt("./y_real.csv", delimiter=",")
x_path = np.loadtxt("./x_path.csv", delimiter=",")
flow_measure = np.loadtxt("./flow_measure.csv", delimiter=",")
x_grid = np.linspace(-1.0, 3.0, 64 + 2)
# visualize consistency of terminal condition
plt.scatter(y_real[:, 48], y_pred[:, 48])
plt.plot(x_grid, flow_measure[:,49])
# visualize the density
