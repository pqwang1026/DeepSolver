import numpy as np

class LinearQuadraticSolver(object):
    def __init__(self, equation, T, Nt, mu0):
        self.equation = equation
        self.T = T
        self.Nt = Nt
        self.h = (T + 0.0) / Nt
        self.k = np.zeros(Nt + 1)
        self.l = np.zeros(Nt + 1)
        self.mu = np.zeros(Nt + 1)
        self.eta = np.zeros(Nt + 1)
        self.xi = np.zeros(Nt + 1)
        self.k[Nt] = equation.q * (equation.q + equation.q_bar)
        self.eta[Nt] = equation.q * equation.q
        self.mu[0] = mu0

    def solveKLEta(self):
        for t in range(self.Nt, 0, -1):
            self.k[t - 1] = -self.h * ((self.k[t] ** 2) * (self.equation.b ** 2) / self.equation.n \
                                       - (2.0 * self.equation.a + self.equation.a_bar) * self.k[t] \
                                        - self.equation.m * (self.equation.m + self.equation.m_bar)) + self.k[t]
            self.l[t - 1] = -self.h * ((self.k[t] * (self.equation.b ** 2) / self.equation.n - self.equation.a) * self.l[t] \
                                        -self.k[t] * self.equation.beta) + self.l[t]
            self.eta[t - 1] = -self.h * ((self.eta[t] ** 2) * (self.equation.b ** 2) / self.equation.n \
                                         - 2.0 * self.equation.a * self.eta[t] - self.equation.m ** 2) + self.eta[t]
    def solveMu(self):
        for t in range(self.Nt):
            self.mu[t + 1] = self.mu[t] + self.h * ((self.equation.a + self.equation.a_bar \
                                                    - (self.equation.b ** 2) * self.k[t] / self.equation.n) * self.mu[t] \
                                                    + self.equation.beta - (self.equation.b ** 2) * self.l[t] / self.equation.n)

    def solveXi(self):
        self.xi[self.Nt] = self.equation.q * self.equation.q_bar * self.mu[self.Nt]
        for t in range(self.Nt, 0, -1):
            self.xi[t - 1] = -self.h * ((self.eta[t] * (self.equation.b ** 2) / self.equation.n - self.equation.a) * self.xi[t] \
                                        -self.equation.m * self.equation.m_bar * self.mu[t] \
                                        -(self.equation.beta + self.equation.a_bar * self.mu[t]) * self.eta[t]) + self.xi[t]
