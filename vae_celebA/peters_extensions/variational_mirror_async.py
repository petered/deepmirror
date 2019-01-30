import os

import sys
from argparse import Namespace
import time
from functools import partial
from multiprocessing import set_start_method

import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from artemis.fileman.file_getter import get_file
from artemis.general.async import iter_latest_asynchonously
from artemis.general.dead_easy_ui import parse_user_function_call
from artemis.general.ezprofile import EZProfiler, profile_context, get_profile_contexts, get_profile_contexts_string
from artemis.general.checkpoint_counter import do_every
from artemis.general.global_rates import limit_rate, limit_iteration_rate
from artemis.general.should_be_builtins import bad_value
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from vae_celebA.dfc_vae import encoder, generator
from vae_celebA.image_utils.face_aligner_2 import FaceAligner2, face_aligning_iterator, display_face_aligner
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.peters_extensions.attention_window import AttentionWindow
from vae_celebA.peters_extensions.fullscreen_display import show_fullscreen, show_fullscreen_v1
from vae_celebA.peters_extensions.hmc_sampler import hmc_leapfrog_step
from vae_celebA.peters_extensions.variational_mirror import crop_by_fraction


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


def momentum_sgd(energy_func, x, v, momentum, step_size):
    gx, = tf.gradients(tf.reduce_sum(energy_func(x)), x)
    v_new = v*momentum - step_size/2. * gx
    x_new = x + v_new
    return x_new, v_new


def multiply_gaussians(means, variances, axis=0, keepdims=True):
    inverse_vars = 1./variances
    new_var = 1./tf.reduce_sum(inverse_vars, axis=axis, keepdims=keepdims)
    new_mean = new_var * tf.reduce_sum(inverse_vars*means, axis=axis, keepdims=keepdims)
    return new_mean, new_var


def crop_by_fraction(im, vcrop, hcrop):
    return im[int(vcrop[0]*im.shape[0]):int(vcrop[1]*im.shape[0]), int(hcrop[0]*im.shape[1]):int(hcrop[1]*im.shape[1])]


def add_fade_frame(img, frame_width=0.05, p_norm=2.):

    r = (np.sum(np.power(np.meshgrid(np.linspace(-1, 1, img.shape[0]), np.linspace(-1, 1, img.shape[1])), p_norm), axis=0))**(1./p_norm)
    fade_mult = (np.minimum(1, np.maximum(0, (1-r)/frame_width)))[:, :, None]
    bordered_image = (img*fade_mult).astype(np.uint8)
    return bordered_image


def correct_gamma(img, gamma = 3.):
    table = (((np.arange(0, 256)/255.0)**(1/gamma))*255).astype(np.uint8)
    return cv2.LUT(img, table)


def equalize_brightness(img, clipLimit=3., tileGridSize=(8, 8)):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def preprocess_image(im, crop_frac = None, gamma = None):

    if crop_frac is not None:
        im = crop_by_fraction(im, *crop_frac)

    if gamma is not None:
        im = correct_gamma(im, gamma=gamma)

    return im


def demo_var_mirror(
        step_size = 0.05,
        video_size = (320, 240),
        momentum_refreshment = 0.2,
        smooth=True,
        display_size=(224, 224),
        show_debug_plots=False,
        show_display_plot=True,
        show_camera_window=True,
        opposite=False,
        v_scale=4.,
        camera_device_no=0,
        display_number=0,
        crop_frac=None,
        multiface_rotation_time = 7,
        display_sizes=[(1440, 900), (1920, 1080)],
        attention_mode = None,
        gamma_correction = 2,
        max_fps = 24,
        async = True,
        fullscreen_version = 0,
        ):

    z_dim = 100
    c_dim=3
    batch_size=1
    output_size=64

    # Setup HMC
    with tf.device("/cpu:0"):
        g = Namespace()
        g.pl_x = tf.placeholder(tf.float32, shape=(1, z_dim))
        g.pl_v = tf.placeholder(tf.float32, shape=(1, z_dim))

        # var_v = tf.placeholder(tf.float32, shape=(1, z_dim))
        g.z_mean = tf.placeholder(tf.float32, shape=(None, z_dim), name='z_mean')
        g.z_var = tf.placeholder(tf.float32, shape=(None, z_dim), name='z_var')

        # Smooth update
        g.x_new, g.v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum(tf.reduce_sum((z-g.z_mean)**2/g.z_var, axis=1), axis=0, keepdims=True), x=g.pl_x, v=g.pl_v, step_size=step_size, momentum_refreshment=momentum_refreshment, v_scale = v_scale*tf.reduce_mean(g.z_var))

        # Random update
        g.z_sample = tf.random_normal(shape=(1, z_dim), mean=g.z_mean, stddev=g.z_var**.5)

        # Setup Generator
        g.z_p = tf.zeros([1, z_dim], tf.float32)
        g.gen0, g.gen0_logits = generator(g.z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False) # reconstruction
        g.full_img = tf.image.resize_images(g.gen0.outputs, display_size)

        # Setup Encoder
        g.input_imgs = tf.placeholder(tf.float32,[None, output_size, output_size, c_dim], name='real_images')
        g.net_out1, g.net_out2, g.qz_mean, g.qz_log_sigma_sq = encoder(g.input_imgs, is_train=True, reuse=False, z_dim=z_dim)
        g.qz_var = tf.exp(g.qz_log_sigma_sq)

        g.qz_mean_prod, g.qz_var_prod = multiply_gaussians(means=g.qz_mean, variances=g.qz_var, axis=0, keepdims=True)

    # Setup Session and load params
    sess = tf.InteractiveSession(config=tf.ConfigProto(device_count=dict(GPU=0)))
    tl.layers.initialize_global_variables(sess)

    gen_file_path = get_file('models/dfc-vae3/net_g.npz', url = 'https://drive.google.com/uc?export=download&id=1YHcctf9l90agJSFFSTiMwjm10WGQO6Lu')
    enc_file_path = get_file('models/dfc-vae3/net_e.npz', url = 'https://drive.google.com/uc?export=download&id=1vLJVU_BDvpDqTfNhC80C3GZI2IxscYwH')

    gen_params = tl.files.load_npz(*os.path.split(gen_file_path))
    tl.files.assign_params(sess, gen_params, g.gen0)

    enc_params = tl.files.load_npz(*os.path.split(enc_file_path))
    tl.files.assign_params(sess, [enc_params[i] for i in range(25)], g.net_out1)
    tl.files.assign_params(sess, [enc_params[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]], g.net_out2)

    # face_detector = get_face_detector_version('video')
    face_detector = FaceAligner2(
        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        desiredFaceWidth=64,
        desiredFaceHeight=64,
    )

    prior_mean = np.zeros([1, z_dim])
    prior_var = np.ones([1, z_dim])
    x = np.random.randn(1, z_dim)
    v = np.random.randn(1, z_dim)*0

    gen_func = partial(
            face_aligning_iterator,
            camera = VideoCamera(size=video_size, device=camera_device_no, hflip=True, mode='rgb'),
            face_aligner=face_detector,
            image_preprocessor=
                ([partial(crop_by_fraction, vcrop=crop_frac[0], hcrop=crop_frac[1])] if crop_frac is not None else []) +
                ([AttentionWindow.from_default_settings(attention_mode)] if attention_mode is not None else []) +
                [partial(correct_gamma, gamma=gamma_correction)]
            )

    iterator = iter_latest_asynchonously(gen_func = gen_func, empty_value=(None, [], []), uninitialized_wait=0.1) if async else gen_func()

    for t, (rgb_im, landmarks, raw_faces) in limit_iteration_rate(enumerate(iterator), period = 1./max_fps):
        with profile_context('total'):
            if rgb_im is None:
                print('No Camera Imageasss')
                time.sleep(0.1)
                continue
            else:
                with profile_context('inference'):
                    if len(raw_faces)>0:
                        faces = raw_faces[[int(time.time()//multiface_rotation_time)%len(raw_faces)]]  # Optional
                        faces = faces/127.5-1.
                        # post_mean, post_var = sess.run([g.qz_mean_prod, g.qz_var_prod], {g.input_imgs: faces})
                        post_mean, post_var = sess.run([g.qz_mean, g.qz_var], {g.input_imgs: faces})
                    else:
                        post_mean, post_var = prior_mean, prior_var

            if opposite:
                post_mean = -post_mean

            if smooth:
                x, v = sess.run((g.x_new, g.v_new), {g.pl_x: x, g.pl_v: v, g.z_mean: post_mean, g.z_var: post_var})
                z_points = x
            else:
                z_points = sess.run(g.z_sample, {g.z_mean: post_mean, g.z_var: post_var})

            with profile_context('generation'):
                im = sess.run(g.full_img, {g.z_p: z_points})

            if show_debug_plots:
                with hold_dbplots():
                    if rgb_im is not None:
                        dbplot(rgb_im, 'im')
                        if len(faces)>0:
                            dbplot(faces, 'faces')
                    dbplot(im, 'image')
                    dbplot(im if rgb_im is None or len(faces)==0 or t%2==0 else faces[0], 'flicker')
            if show_display_plot:
                face_img = ((im[0, :, :, ::-1]+1.)*127.5).astype(np.uint8)

                if fullscreen_version == 0:
                    show_fullscreen(image = face_img, background_colour=(0, 0, 0), display_sizes=display_sizes, display_number=display_number)
                else:
                    assert fullscreen_version == 1
                    show_fullscreen_v1(image = face_img, background_colour=(0, 0, 0), display_sizes=display_sizes, display_number=display_number)
            if show_camera_window:
                # print(rgb_im.shape)
                display_face_aligner(rgb_im=rgb_im, landmarks=landmarks, faces=raw_faces)

        if do_every('5s'):
            print(get_profile_contexts_string(['total', 'generation', 'inference'], fill_empty_with_zero=True))


if __name__ == '__main__':

    _, args, kwargs = parse_user_function_call(' '.join(sys.argv))

    args = sys.argv[1:]

    mode = 'lab' if len(args)==0 else args[0] if len(args)==1 else bad_value(f'Can only provided 1 unnamed arg, for mode.  You provided: {args}')

    set_start_method('forkserver', force=True)

    print(f'Running Mode: {mode}')
    if mode == 'laptop':
        keyword_args = dict(video_size = (320, 240), display_sizes=[(1440, 900), (1920, 1080)], fullscreen_version = 0)
    elif mode == 'lab':
        keyword_args = dict(video_size = (320, 240), display_sizes=[(1600, 1200)], fullscreen_version = 0)
    elif mode == 'outside':
        keyword_args = dict(video_size = (640, 480), attention_mode = 'faces', display_sizes=[(1440, 900), (1920, 1080)])
    elif mode == 'box-outside':
        keyword_args = dict(video_size = (640, 480), attention_mode = 'faces', display_sizes=[(1920, 1080)], fullscreen_version=1)
    else:
        raise NotImplementedError(mode)

    keyword_args.update(kwargs)

    demo_var_mirror(**keyword_args)
