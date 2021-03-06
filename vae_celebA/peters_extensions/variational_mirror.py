import itertools
import os
from argparse import Namespace
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from artemis.fileman.file_getter import get_file
from artemis.general.ezprofile import EZProfiler, profile_context, get_profile_contexts
from artemis.general.checkpoint_counter import do_every
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from vae_celebA.dfc_vae import encoder, generator
from vae_celebA.image_utils.face_aligner_2 import FaceAligner2
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.peters_extensions.fullscreen_display import show_fullscreen, show_fullscreen_v1
from vae_celebA.peters_extensions.hmc_sampler import hmc_leapfrog_step


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


def demo_var_mirror(
        n_steps=None,
        step_size = 0.05,
        video_size = (320, 240),
        momentum_refreshment = 0.2,
        smooth=True,
        display_size=(224, 224),
        show_debug_plots=False,
        show_display_plot=False,
        show_camera_window=False,
        opposite=False,
        v_scale=4.,
        camera_device_no=0,
        display_number=0,
        crop_frac=None,
        display_sizes=[(1440, 900), (1920, 1080)],
        do_brightness_equalization = True,
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
        # g.x_new, g.v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum((z-g.z_mean)**2/g.z_var, axis=1), x=g.pl_x, v=g.pl_v, step_size=step_size, momentum_refreshment=momentum_refreshment, v_scale = v_scale*tf.reduce_mean(g.z_var))
        g.x_new, g.v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum(tf.reduce_sum((z-g.z_mean)**2/g.z_var, axis=1), axis=0, keepdims=True), x=g.pl_x, v=g.pl_v, step_size=step_size, momentum_refreshment=momentum_refreshment, v_scale = v_scale*tf.reduce_mean(g.z_var))
        # g.x_new, g.v_new = momentum_sgd(lambda z: 0.5 * tf.reduce_sum((z-g.z_mean)**2/g.z_var, axis=1), x=pl_x, v=pl_v, step_size=0.01, momentum=0.9)

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
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    gen_file_path = get_file('models/dfc-vae3/net_g.npz', url = 'https://drive.google.com/uc?export=download&id=1YHcctf9l90agJSFFSTiMwjm10WGQO6Lu')
    enc_file_path = get_file('models/dfc-vae3/net_e.npz', url = 'https://drive.google.com/uc?export=download&id=1vLJVU_BDvpDqTfNhC80C3GZI2IxscYwH')
    # LOCAL = True
    # if LOCAL:
    #     netpath = get_file('models/dfc-vae3', url = )
    # else:
    #     netpath = '/home/petered/remote/vae-celebA/vae_celebA/checkpoint/dfc-vae3'

    gen_params = tl.files.load_npz(*os.path.split(gen_file_path))
    tl.files.assign_params(sess, gen_params, g.gen0)

    enc_params = tl.files.load_npz(*os.path.split(enc_file_path))
    tl.files.assign_params(sess, [enc_params[i] for i in range(25)], g.net_out1)
    tl.files.assign_params(sess, [enc_params[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]], g.net_out2)

    # face_detector = get_face_detector_version('video')
    face_detector = FaceAligner2(
        # desiredLeftEye = (0.38, 0.50),

        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        # desiredLeftEye = (0.39, 0.51),
        desiredFaceWidth=64,
        desiredFaceHeight=64,
    )

    # cam = VideoCamera(size=(640, 480))
    cam = VideoCamera(size=video_size, device=camera_device_no, hflip=True, mode='rgb')
    prior_mean = np.zeros([1, z_dim])
    prior_var = np.ones([1, z_dim])
    x = np.random.randn(1, z_dim)
    v = np.random.randn(1, z_dim)*0

    # Run
    # im = np.zeros((64, 64, 3))
    for t in range(n_steps) if n_steps is not None else itertools.count(0):

        with profile_context('total'):
            rgb_im = cam.read()

            if rgb_im is None:
                print('No Camera Image')
                time.sleep(0.1)
                continue

            if rgb_im is not None:

                if crop_frac is not None:
                    rgb_im = crop_by_fraction(rgb_im, *crop_frac)

                if do_brightness_equalization:
                    rgb_im = correct_gamma(rgb_im)

                # rgb_im = bgr_im[..., ::-1, ::-1]  # Flip for mirror effect
                # rgb_im = bgr_im[..., ::-1, :]  # Flip for mirror effect

                with profile_context('detection'):
                    # faces = face_detector(rgb_im)
                    landmarks, raw_faces = face_detector(rgb_im)

                with profile_context('inference'):
                    if len(raw_faces)>0:
                        faces = raw_faces[[int(time.time()//10)%len(raw_faces)]]  # Optional
                        faces = faces/127.5-1.
                        # post_mean, post_var = sess.run([g.qz_mean_prod, g.qz_var_prod], {g.input_imgs: faces})
                        post_mean, post_var = sess.run([g.qz_mean, g.qz_var], {g.input_imgs: faces})
                    else:
                        post_mean, post_var = prior_mean, prior_var
            else:
                landmarks, faces = [], []
                print('No Camera Image!')

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
                # face_img=add_fade_frame(face_img, p_norm=1.2)

                # cv2.imshow('face_img', face_img)

                # show_fullscreen_v1(image = face_img, background_colour=(0, 0, 0), display_sizes=display_sizes, display_number=display_number)
                show_fullscreen(image = face_img, background_colour=(0, 0, 0), display_sizes=display_sizes, display_number=display_number)
            if show_camera_window:
                display_img = rgb_im[..., ::-1].copy()

                for landmark, face in zip(landmarks, raw_faces):
                    cv2.circle(display_img, tuple(landmark.left_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
                    cv2.circle(display_img, tuple(landmark.right_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
                    display_img[-face.shape[0]:, -face.shape[1]:, ::-1] = face

                cv2.imshow('camera', display_img)
                cv2.waitKey(1)
        if do_every('5s'):
            profile = get_profile_contexts(['total', 'detection', 'generation', 'inference'], fill_empty_with_zero=True)
            print(f'Mean Times:: Total: {profile["total"][1]/profile["total"][0]:.3g}, Detection: {profile["detection"][1]/profile["detection"][0]:.3g}, Inference: {profile["inference"][1]/profile["inference"][0]:.3g}, Generation: {profile["generation"][1]/profile["generation"][0]:.3g}')


if __name__ == '__main__':

    OUTSIDE = False
    display_sizes = [(1440, 900), (1920, 1080)]

    if OUTSIDE:
        crop_frac = [(.3, .7), (0, 1)]
        video_size = (640, 480)
    else:
        crop_frac = None
        video_size = (320, 240)

    demo_var_mirror(
        smooth=True,
        opposite = False,
        show_debug_plots=False,
        show_display_plot=True,
        show_camera_window=True,
        camera_device_no=0,
        video_size = video_size,
        crop_frac=crop_frac,
        display_number=0,
        display_sizes = display_sizes,
        do_brightness_equalization=True
        )
