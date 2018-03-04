import time
import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
from scipy.stats import multivariate_normal as normal
from tensorflow.python.ops import control_flow_ops
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init


class SolverMKV1:
    def __init__(self, sess, equation, T, Nt, NDirac, x0_mean, x0_var):
        # initial condition and horizon
        self.x0_mean = x0_mean
        self.x0_var = x0_var
        self.T = T

        # tensorflow session
        self.sess = sess

        # equation (of mkvequation class)
        self.equation = equation

        # parameters for construct the grid of time and grid of space
        self.Nx = equation.Nx
        self.Ny = equation.Ny
        self.Nt = Nt

        self.NDirac = NDirac  # bandwidth of gaussian density approximating Dirac measure
        self.dx = equation.dx
        self.dy = equation.dy
        self.h = (T + 0.0) / Nt
        self.sqrth = np.sqrt(self.h)
        self.t_grid = tf.constant(np.linspace(0, T, Nt + 1), dtype=tf.float64, shape=(Nt + 1, 1))
        self.x_grid = equation.x_grid
        self.y_grid = equation.y_grid
        self.x_grid_2d = tf.matmul(self.x_grid, tf.ones(shape=(1, (self.Ny + 2)), dtype=tf.float64))
        self.y_grid_2d = tf.matmul(tf.ones(shape=((self.Nx + 2), 1), dtype=tf.float64), self.y_grid, transpose_b=True)

        # parameters for neural network and gradient descent
        self.n_layer = 4
        self.n_neuron = [1, 1 + 10, 1 + 10, 1]
        self.batch_size = 64
        self.valid_size = 256
        self.n_maxstep = 16000
        self.n_displaystep = 50
        self.learning_rate = 5e-4
        self._extra_train_ops = []


        # useful tensors
        self.Px0 = tf.exp(-(self.x_grid-self.x0_mean) ** 2 / 2 / self.x0_var) / np.sqrt(2.0 * np.pi * self.x0_var)
        self.dW = tf.placeholder(tf.float64, [None, self.Nt], name='dW')
        self.X0 = tf.placeholder(tf.float64, [None, 1], name = 'X0')
        self.concat_operator_grid = tf.placeholder(tf.float64, [None, None], name = 'ConcatOperator1')
        self.concat_operator_sample = tf.placeholder(tf.float64, [None, None], name = 'ConcatOperator2')
        self.initial_loss = tf.constant([0], dtype=tf.float64, shape=[1])
        self.is_training = tf.placeholder(tf.bool)

        # timing
        self.time_build = 0

    def sample_noise(self, n_sample):
        dW_sample = np.zeros([n_sample, self.Nt])
        for i in range(self.Nt):
            dW_sample[:, i] = normal.rvs(mean=0.0, cov=1.0, size=n_sample) * self.sqrth * self.equation.sigma
        return dW_sample

    def sample_x0(self, n_sample):
        X0_sample = np.zeros([n_sample,1])
        X0_sample[:,0] = normal.rvs(mean=self.x0_mean, cov=self.x0_var, size=n_sample)
        return X0_sample

    def _one_time_net(self, x, name):
        with tf.variable_scope(name):
            layer1 = self._one_layer(x, self.n_neuron[0], self.n_neuron[1], name='layer1')
            layer2 = self._one_layer(layer1, self.n_neuron[1], self.n_neuron[2], name='layer2')
            z = self._one_layer(layer2, self.n_neuron[2], self.n_neuron[3], activation_fn=None, name='final')
            return z

            # one layer in the neural network

    def _one_layer(self, input_, in_sz, out_sz, activation_fn=tf.nn.relu, std=5.0, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('Weight', [in_sz, out_sz], tf.float64,
                                norm_init(stddev=std / np.sqrt(in_sz + out_sz)))
            hidden = tf.matmul(input_, w)
        if activation_fn is not None:
            return activation_fn(hidden)
        else:
            return hidden

    def generate_dict_debug(self):
        dW_debug = self.sample_noise(17)
        X0_debug = self.sample_x0(17)
        return {self.dW: dW_debug, self.X0: X0_debug, self.is_training: False}

    def build(self):
        start_time = time.time()
        # build the dynamic of MKV-SDE
        self.loss = self.initial_loss
        X = self.X0
        X_all = tf.concat([self.x_grid, X], axis=0)
        phi_X_all = self._one_time_net(X_all, 'decoupling_field'+str(0))
        phi_prime_X_all = self.equation.sigma * tf.gradients(phi_X_all, X_all)[0]
        Y_kolmogorov = phi_X_all[0:(self.Nx + 2),:]
        Z_kolmogorov = phi_prime_X_all[0:(self.Nx + 2),:]
        phi_X = phi_X_all[(self.Nx + 2):,:]
        phi_prime_X = phi_prime_X_all[(self.Nx + 2):,:]
        P = self.Px0
        self.flow_of_measure = P
        self.Y_guess_full = phi_X
        self.Y_real_full = phi_X
        X_next = X + self.equation.drift_b(self.t_grid[0], X, phi_X, phi_prime_X, P) * self.h + self.dW[:, 0:1]
        Y_next = phi_X - self.equation.driver_f(self.t_grid[0], X, phi_X, phi_prime_X, P) * self.h \
                 + phi_prime_X * self.dW[:, 0:1]
        P_reduced = tf.slice(P, [1, 0], [self.Nx, 1])
        P_next_reduced = P_reduced + self.equation.kolmogorov_1d(self.t_grid[0], self.x_grid, Y_kolmogorov, Z_kolmogorov, P, self.dx, self.Nx) * self.h
        P = tf.concat([tf.slice(P_next_reduced, [0, 0], [1, 1]), P_next_reduced, tf.slice(P_next_reduced, [self.Nx - 1, 0], [1, 1])], axis=0)
        X = X_next
        Y = Y_next
        for t in range(1, self.Nt - 1):
            print(t)
            X_all = tf.concat([self.x_grid, X], axis=0)
            phi_X_all = self._one_time_net(X_all, 'decoupling_field'+str(t))
            phi_prime_X_all = tf.gradients(phi_X_all, X_all)[0]
            phi_X_kolmogorov = phi_X_all[0:(self.Nx + 2), :]
            phi_prime_X_kolmogorov = phi_prime_X_all[0:(self.Nx + 2), :]
            phi_X = phi_X_all[(self.Nx + 2):, :]
            self.Y_guess_full = tf.concat([self.Y_guess_full, phi_X], axis=1)
            self.Y_real_full = tf.concat([self.Y_real_full, Y], axis=1)
            self.flow_of_measure = tf.concat([self.flow_of_measure, P], axis=1)
            error = Y - phi_X
            self.loss = self.loss + tf.reduce_mean(error ** 2)
            phi_prime_X = phi_prime_X_all[(self.Nx + 2):, :]
            P = self.Px0
            X_next = X + self.equation.drift_b(self.t_grid[0], X, phi_X, phi_prime_X, P) * self.h + self.dW[:, 0:1]
            Y_next = phi_X - self.equation.driver_f(self.t_grid[0], X, phi_X, phi_prime_X, P) * self.h \
                     + phi_prime_X * self.dW[:, 0:1]
            P_reduced = tf.slice(P, [1, 0], [self.Nx, 1])
            P_next_reduced = P_reduced + \
                             self.equation.kolmogorov_1d(self.t_grid[0], self.x_grid, phi_X_kolmogorov,
                                                         phi_prime_X_kolmogorov, P, self.dx, self.Nx) * self.h
            P = tf.concat([tf.slice(P_next_reduced, [0, 0], [1, 1]), P_next_reduced, tf.slice(P_next_reduced, [self.Nx - 1, 0], [1, 1])], axis=0)
            X = X_next
            Y = Y_next
        target = self.equation.terminal_g(X, P)
        error = Y - target
        self.Y_guess_full = tf.concat([self.Y_guess_full, target], axis=1)
        self.Y_real_full = tf.concat([self.Y_real_full, Y], axis=1)
        self.flow_of_measure = tf.concat([self.flow_of_measure, P], axis=1)
        error_terminal_rel = tf.abs(error / target)
        self.loss = self.loss + tf.reduce_mean(error ** 2)
        self.loss = self.loss / self.Nt
        self.loss_terminal_rel = tf.reduce_mean(error_terminal_rel)
        self.time_build = time.time() - start_time
        print("Build complete build time: %4u" % self.time_build)

    def train(self):
        start_time = time.time()
        # train operations
        self.global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(1),
                                           trainable = False, dtype = tf.int32)
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self.loss_history = []
        self.init_history = []
        dW_valid = self.sample_noise(self.valid_size)
        X0_valid = self.sample_x0(self.valid_size)
        feed_dict_valid = {self.dW: dW_valid, self.X0: X0_valid, self.is_training: False}

        # initialization
        step = 1
        self.sess.run(tf.global_variables_initializer())
        temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
        terminal_loss_rel = self.sess.run(self.loss_terminal_rel, feed_dict=feed_dict_valid) * 100.0
        self.loss_history.append(temp_loss)
        print("step: %5u, loss: %.4e, relative terminal loss : %.4e %%" % (0, temp_loss, terminal_loss_rel) + \
              "runtime: %4u" % (time.time() - start_time + self.time_build))

        # begin sgd iteration
        for _ in range(self.n_maxstep + 1):
            step = self.sess.run(self.global_step)
            dW_train = self.sample_noise(self.batch_size)
            X0_train = self.sample_x0(self.batch_size)
            feed_dict_train = {self.dW: dW_train, self.X0: X0_train, self.is_training: False}
            self.sess.run(self.train_op, feed_dict=feed_dict_train)
            if step % self.n_displaystep == 0:
                temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
                terminal_loss_rel = self.sess.run(self.loss_terminal_rel, feed_dict=feed_dict_valid)
                self.loss_history.append(temp_loss)
                print("step: %5u, loss: %.4e, relative terminal loss : %.4e " % (step, temp_loss, terminal_loss_rel) + \
                    "runtime: %4u" % (time.time() - start_time + self.time_build))

            step += 1
        self.y_real = self.sess.run(self.Y_real_full, feed_dict=feed_dict_valid)
        self.y_pred = self.sess.run(self.Y_guess_full, feed_dict=feed_dict_valid)
        self.flow_measure = self.sess.run(self.flow_of_measure, feed_dict=feed_dict_valid)
        end_time = time.time()
        print("running time: %.3f s" % (end_time - start_time + self.time_build))
