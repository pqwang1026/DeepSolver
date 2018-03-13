import mkvequation
import mkvsolver1
import tensorflow as tf
import numpy as np

def main():
    # parameters
    LxUpper = 3.0
    LxLower = -1.0
    Nx = 64
    T = 0.5
    Nt = 50
    x0_mean = 1.0
    x0_var = 0.1
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(1)
        print("Begin to solve MKV FBSDE")
        mkv_equation = mkvequation.MckeanVlasovEquationLQMFG(LxUpper, LxLower, Nx)
        mkv_solver = mkvsolver1.SolverMKV1(sess, mkv_equation, T, Nt, x0_mean, x0_var)
        mkv_solver.build()
        mkv_solver.train()
        np.savetxt("./y_pred2.csv", mkv_solver.y_pred, fmt='%.5e', delimiter=",")
        np.savetxt("./y_real2.csv", mkv_solver.y_real, fmt='%.5e', delimiter=",")
        np.savetxt("./x_path2.csv", mkv_solver.x_path, fmt='%.5e', delimiter=",")
        np.savetxt("./flow_measure2.csv", mkv_solver.flow_measure, fmt='%.5e', delimiter=",")

if __name__ == '__main__':
    np.random.seed(1)
    main()