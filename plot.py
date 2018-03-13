import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lqsolver
import mkvequation

# solve LQMFG using ODE
LxUpper = 3.0
LxLower = -1.0
Nx = 64
T = 0.5
Nt = 50
mu0 = 1.0
mkv_equation = mkvequation.MckeanVlasovEquationLQMFG(LxUpper, LxLower, Nx)
odeSolver =lqsolver.LinearQuadraticSolver(mkv_equation, T, Nt, mu0)
odeSolver.solveKLEta()
odeSolver.solveMu()
odeSolver.solveXi()

# read solution from deep solver
x_path = np.loadtxt("./x_path2.csv", delimiter=",")
flow_measure = np.loadtxt("./flow_measure2.csv", delimiter=",")
x_grid = np.linspace(-1.0, 3.0, 64 + 2)
y_pred = np.loadtxt("./y_pred2.csv", delimiter=",")
y_real = np.loadtxt("./y_real2.csv", delimiter=",")

# visualize the consistency of the solution
y_pred_flat = np.reshape(y_pred, [51*128])
y_real_flat = np.reshape(y_real, [51*128])
plt.scatter(y_real_flat, y_pred_flat, s=2)

# visualize and compare the mean of x
x_mean = np.sum(flow_measure * np.outer(x_grid, np.ones(51)), axis = 0) * (x_grid[1] - x_grid[0])
fig, ax = plt.subplots()
timeline = np.linspace(0, T, Nt + 1)
ax.margins(0.05)
ax.plot(timeline, x_mean, label="Deep Solver")
ax.plot(timeline, odeSolver.mu, label="ODE")
ax.legend()
plt.show()

# visualize and compare the decoupling field
t = 48
x_min = np.min(x_path[:, t])
x_max = np.max(x_path[:, t])
x_line = np.linspace(x_min, x_max, 50)
y_line = x_line * odeSolver.eta[t] + odeSolver.xi[t]
fig, ax = plt.subplots()
ax.margins(0.05)
ax.plot(x_line, y_line, color="red", label="ODE")
ax.scatter(x_path[:, t], y_pred[:, t], s=10, color="blue", label="Deep Solver")
ax.legend()
plt.show()

# visualize consistency of terminal condition
plt.plot(x_grid, flow_measure[:,49])

# visualize the density
fig, ax = plt.subplots()
ax.margins(0.05)
ax.plot(x_grid, flow_measure[:,0], label='t=0')
ax.plot(x_grid, flow_measure[:,10], label='t=0.1')
ax.plot(x_grid, flow_measure[:,20], label='t=0.2')
ax.plot(x_grid, flow_measure[:,30], label='t=0.3')
ax.plot(x_grid, flow_measure[:,40], label='t=0.4')
ax.plot(x_grid, flow_measure[:,49], label='t=0.5')
ax.legend()
plt.show()