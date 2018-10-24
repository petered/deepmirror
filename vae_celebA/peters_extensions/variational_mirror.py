import numpy as np
import tensorflow as tf
import tensorlayer as tl

from artemis.fileman.file_getter import get_artemis_data_path
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from vae_celebA.dfc_vae import encoder, generator
from vae_celebA.image_utils.face_aligner import FaceAligner
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.peters_extensions.hmc_sampler import hmc_leapfrog_step
import cv2
import itertools


def demo_decoder(n_steps=1000, step_size = 0.1, momentum_refreshment = 0.1):

    z_dim = 100
    c_dim=3
    batch_size=1
    output_size=64

    # Setup HMC
    x = np.random.randn(1, z_dim)
    v = np.random.randn(1, z_dim)*0
    var_x = tf.placeholder(tf.float32, shape=(1, z_dim))
    var_v = tf.placeholder(tf.float32, shape=(1, z_dim))
    x_new, v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum(z**2, axis=1), x=var_x, v=var_v, step_size=step_size, momentum_refreshment=momentum_refreshment)

    # Setup Generator
    z_p = tf.zeros([1, z_dim], tf.float32)
    gen0, gen0_logits = generator(z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False) # reconstruction
    full_img = tf.image.resize_images(gen0.outputs,[224,224])

    # Setup Session and load params
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    params = tl.files.load_npz('/home/petered/remote/vae-celebA/vae_celebA/checkpoint/dfc-vae3', 'net_g.npz')
    tl.files.assign_params(sess, params, gen0)

    # Run
    for t in range(n_steps):
        x, v = sess.run((x_new, v_new), {var_x: x, var_v: v})
        im = sess.run(full_img, {z_p: x})
        dbplot(im, 'image')


def demo_var_mirror(n_steps=None, step_size = 0.05, momentum_refreshment = 0.1, smooth=True, display_size=(224, 224), show_debug_plots=False):

    z_dim = 100
    c_dim=3
    batch_size=1
    output_size=64

    # Setup HMC
    pl_x = tf.placeholder(tf.float32, shape=(1, z_dim))
    pl_v = tf.placeholder(tf.float32, shape=(1, z_dim))

    # var_v = tf.placeholder(tf.float32, shape=(1, z_dim))
    z_mean = tf.placeholder(tf.float32, shape=(1, z_dim), name='z_mean')
    z_var = tf.placeholder(tf.float32, shape=(1, z_dim), name='z_var')

    # Smooth update
    x_new, v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum((z-z_mean)**2/z_var, axis=1), x=pl_x, v=pl_v, step_size=step_size, momentum_refreshment=momentum_refreshment)

    # Random update
    z_sample = tf.random_normal(shape=(1, z_dim), mean=z_mean, stddev=z_var**.5)

    # Setup Generator
    z_p = tf.zeros([1, z_dim], tf.float32)
    gen0, gen0_logits = generator(z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False) # reconstruction
    full_img = tf.image.resize_images(gen0.outputs, display_size)

    # Setup Encoder
    input_imgs = tf.placeholder(tf.float32,[batch_size, output_size, output_size, c_dim], name='real_images')
    net_out1, net_out2, qz_mean, qz_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False, z_dim=z_dim)
    qz_var = tf.exp(qz_log_sigma_sq)

    # Setup Session and load params
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    LOCAL = True
    if LOCAL:
        netpath = get_artemis_data_path('models/dfc-vae3')
    else:
        netpath = '/home/petered/remote/vae-celebA/vae_celebA/checkpoint/dfc-vae3'

    gen_params = tl.files.load_npz(netpath, 'net_g.npz')
    tl.files.assign_params(sess, gen_params, gen0)

    enc_params = tl.files.load_npz(netpath, 'net_e.npz')
    tl.files.assign_params(sess, [enc_params[i] for i in range(25)], net_out1)
    tl.files.assign_params(sess, [enc_params[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]], net_out2)

    # face_detector = get_face_detector_version('video')
    face_detector = FaceAligner.build_default(
        desiredLeftEye = (0.38, 0.50),
        # desiredLeftEye = (0.39, 0.51),
        desiredFaceWidth=64,
        desiredFaceHeight=64,
    )

    cam = VideoCamera(size=(640, 480))
    prior_mean = np.zeros([1, z_dim])
    prior_var = np.ones([1, z_dim])
    x = np.random.randn(1, z_dim)
    v = np.random.randn(1, z_dim)*0

    # Run
    # im = np.zeros((64, 64, 3))
    for t in range(n_steps) if n_steps is not None else itertools.count(0):
        bgr_im = cam.read()

        if bgr_im is not None:
            rgb_im = bgr_im[..., ::-1, ::-1]  # Flip for mirror effect

            faces = face_detector(rgb_im)

            if len(faces)>0:
                faces = faces/127.5-1.
                post_mean, post_var = sess.run([qz_mean, qz_var], {input_imgs: faces})
            else:
                post_mean, post_var = prior_mean, prior_var
        else:
            post_mean, post_var = prior_mean, prior_var

        if smooth:
            x, v = sess.run((x_new, v_new), {pl_x: x, pl_v: v, z_mean: post_mean, z_var: post_var})
            z_points = x
        else:
            z_points = sess.run(z_sample, {z_mean: post_mean, z_var: post_var})

        im = sess.run(full_img, {z_p: z_points})

        if show_debug_plots:
            with hold_dbplots():
                if bgr_im is not None:
                    dbplot(rgb_im, 'im')
                    if len(faces)>0:
                        dbplot(faces, 'faces')
                dbplot(im, 'image')
                dbplot(im if bgr_im is None or len(faces)==0 or t%2==0 else faces[0], 'flicker')
        else:
            cv2.imshow('mirror', cv2.resize(((im[0, :, :, ::-1]+1.)*127.5).astype(np.uint8), dsize=(800, 800)))
            cv2.waitKey(1)


if __name__ == '__main__':
    # demo_decoder()
    demo_var_mirror()