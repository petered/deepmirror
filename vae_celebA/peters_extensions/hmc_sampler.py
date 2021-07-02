import tensorflow as tf
import numpy as np

from artemis.plotting.db_plotting import dbplot


def hmc_leapfrog_step(energy_func, x, v, step_size=0.1, momentum_refreshment = 0., v_scale = 1.):
    """
    Note: incomplete as an MCMC sampler because there's no rejection step'
    See reference http://www.mcmchandbook.net/HandbookChapter5.pdf Equations 5.18-5.20

    :param energy_func: A function mapping an (n_samples, ...) input to an (n_samples, ) vector of energies
    :param x: A state vector
    :param v: A velocity vector
    :return: x, v: Updated state and velocity.
    """
    gx, = tf.gradients(tf.reduce_sum(energy_func(x)), x)
    v_half = v - step_size/2. * gx
    x_new = x + step_size * v_half
    gx, = tf.gradients(tf.reduce_sum(energy_func(x_new)), x_new)
    v_new = v_half - step_size/2. * gx

    v_new = (1-momentum_refreshment)*v_new + momentum_refreshment * tf.random_normal(shape=v.shape) * v_scale

    return x_new, v_new



def demo_hmc():

    energy_func = lambda x: tf.reduce_sum(x**2/2, axis=1)

    sess = tf.Session()

    x = np.random.randn(1, 2)
    v = np.random.randn(1, 2)*0
    step_size = 0.1
    n_steps = 10000
    momentum_refreshment = 0.1

    var_x = tf.placeholder(tf.float32, shape=(1, 2))
    var_v = tf.placeholder(tf.float32, shape=(1, 2))

    x_new, v_new = hmc_leapfrog_step(energy_func, x=var_x, v=var_v, step_size=step_size, momentum_refreshment=momentum_refreshment)

    for t in range(n_steps):

        x, v = sess.run((x_new, v_new), {var_x: x, var_v: v})

        # x = x + np.random.randn(*x.shape) - .1*x

        dbplot((x[0, 0], x[0, 1]), 'trajectory', draw_every=5)


if __name__ == '__main__':
    demo_hmc()



