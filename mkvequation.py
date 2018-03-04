import tensorflow as tf
import numpy as np


class MckeanVlasovEquation(object):
    def __init__(self, LxUpper, LyUpper, LxLower, LyLower, Nx, Ny):
        self.sigma = 0.2
        self.Nx = Nx
        self.Ny = Ny
        self.x_grid = tf.constant(np.linspace(LxLower, LxUpper, Nx + 2), dtype=tf.float64, shape=(Nx + 2, 1))
        self.y_grid = tf.constant(np.linspace(LyLower, LyUpper, Ny + 2), dtype=tf.float64, shape=(Ny + 2, 1))
        self.dx = (LxUpper - LxLower) / (Nx + 1)
        self.dy = (LyUpper - LyLower) / (Ny + 1)

    def marginal_density_x(self, P):
        return self.dy * tf.reduce_sum(P, axis=1, keepdims=True)

    def expectation_x(self, P):
        px = self.marginal_density_x(P)
        return self.dx * tf.reduce_sum(px * self.x_grid)

    def second_moment_x(self, P):
        px = self.marginal_density_x(P)
        return self.dx * tf.reduce_sum(px * (self.x_grid ** 2))


    def driver_f(self, t, X, Y, Z, P):
        return tf.atan(self.expectation_x(P))

    def drift_b(self, t, X, Y, Z, P):
        return -Y

    def terminal_g(self, X, P):
        return tf.atan(X)

    def kolmogorov_1d(self, t, x_grid_1d, y_grid_1d, z_grid_1d, P, dx, Nx):
        term1 = self.drift_b(t, x_grid_1d[2:(Nx + 2), :], y_grid_1d[2:(Nx + 2), :], z_grid_1d[2:(Nx + 2), :], P) * P[2:(Nx + 2), :]\
                - self.drift_b(t, x_grid_1d[1:(Nx + 1), :], y_grid_1d[1:(Nx + 1), :], z_grid_1d[1:(Nx + 1), :], P) * P[1:(Nx + 1), :]
        term1 = tf.scalar_mul(-1.0/dx, term1)
        term2 = P[2:(Nx + 2), :] + P[0:Nx, :] - 2.0 * P[1:(Nx + 1), :]
        term2 = tf.scalar_mul(0.25 * self.sigma * self.sigma / dx / dx, term2)
        return term1 + term2

    def kolmogorov_2d(self, t, x_grid_2d, y_grid_2d, z_grid_2d, P, dx, dy, Nx, Ny):
        term1 = self.drift_b(t, x_grid_2d[2:(Nx + 2), 1:(Ny + 1)], y_grid_2d[2:(Nx + 2), 1:(Ny + 1)], z_grid_2d[2:(Nx + 2), 1:(Ny + 1)], P) * P[2:(Nx + 2), 1:(Ny + 1)]\
                - self.drift_b(t, x_grid_2d[1:(Nx + 1), 1:(Ny + 1)], y_grid_2d[1:(Nx + 1), 1:(Ny + 1)], z_grid_2d[1:(Nx + 1), 1:(Ny + 1)], P) * P[1:(Nx + 1), 1:(Ny + 1)]
        term1 = tf.scalar_mul(-1.0/dx, term1)
        term2 = self.driver_f(t, x_grid_2d[1:(Nx + 1), 2:(Ny + 2)], y_grid_2d[1:(Nx + 1), 2:(Ny + 2)], z_grid_2d[1:(Nx + 1), 2:(Ny + 2)], P) * P[1:(Nx + 1), 2:(Ny + 2)]\
                - self.driver_f(t, x_grid_2d[1:(Nx + 1), 1:(Ny + 1)], y_grid_2d[1:(Nx + 1), 1:(Ny + 1)], z_grid_2d[1:(Nx + 1), 1:(Ny + 1)], P) * P[1:(Nx + 1), 1:(Ny + 1)]
        term2 = tf.scalar_mul(1.0/dy, term2)
        term3 = P[2:(Nx + 2), 1:(Ny + 1)] + P[0:Nx, 1:(Ny + 1)] - 2.0 * P[1:(Nx + 1), 1:(Ny + 1)]
        term3 = tf.scalar_mul(0.25 * self.sigma * self.sigma / dx / dx, term3)
        term4 = (z_grid_2d[1:(Nx + 1), 1:(Ny + 1)] ** 2) * (P[1:(Nx + 1), 2:(Ny + 2)] + P[1:(Nx + 1), 0:Ny] - P[1:(Nx + 1), 1:(Ny + 1)])
        term4 = tf.scalar_mul(0.25 / dy / dy, term4)
        term5 = z_grid_2d[2:(Nx + 2), 1:(Ny + 1)] * P[2:(Nx + 2), 2:(Ny + 2)] + z_grid_2d[0:Nx, 1:(Ny + 1)] * P[0:Nx, 0:Ny]\
                - z_grid_2d[2:(Nx + 2), 1:(Ny + 1)] * P[2:(Nx + 2), 0:Ny] - z_grid_2d[0:Nx, 1:(Ny + 1)] * P[0:Nx, 2:(Ny + 2)]
        term5 = tf.scalar_mul(0.25 * self.sigma / dx / dy, term5)
        return term1 + term2 + term3 + term4 + term5


