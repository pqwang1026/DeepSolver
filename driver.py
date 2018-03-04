import mkvequation
import mkvsolver1
import tensorflow as tf
import numpy as np

def main():
    # parameters
    LxUpper = 2.5
    LxLower = -0.5
    LyUpper = 0.0
    LyLower = -0.5
    Nx = 60
    Ny = 50
    NDirac = 100.0
    T = 2.0
    Nt = 200
    x0_mean = 1.0
    x0_var = 0.1
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(1)
        print("Begin to solve MKV FBSDE")
        mkv_equation = mkvequation.MckeanVlasovEquation(LxUpper, LyUpper, LxLower, LyLower, Nx, Ny)
        mkv_solver = mkvsolver1.SolverMKV1(sess, mkv_equation, T, Nt, NDirac, x0_mean, x0_var)
        mkv_solver.build()
        mkv_solver.train()
        np.savetxt("./y_pred.csv", mkv_solver.y_pred, fmt='%.5e', delimiter=",")
        np.savetxt("./y_real.csv", mkv_solver.y_real, fmt='%.5e', delimiter=",")
        np.savetxt("./flow_measure.csv", mkv_solver.flow_measure, fmt='%.5e', delimiter=",")

if __name__ == '__main__':
    np.random.seed(1)
    main()