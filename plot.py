import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y_pred = np.loadtxt("./y_pred.csv", delimiter=",")
    y_real = np.loadtxt("./y_real.csv", delimiter=",")
    flow_measure = np.loadtxt("./flow_measure.csv", delimiter=",")
    x_grid = np.linspace(-0.5, 2.0, 50 + 2)
    # visualize consistency of terminal condition
    plt.scatter(y_real[:, 49], y_pred[:, 49])
    plt.plot(x_grid, flow_measure[:,49])
    # visualize the density
