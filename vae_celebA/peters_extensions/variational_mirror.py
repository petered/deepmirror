import itertools
import os
from argparse import Namespace
import time
from collections import OrderedDict
from enum import Enum
from typing import Tuple, Dict, Mapping, Sequence, Iterable, Callable, Optional

import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from artemis.fileman.file_getter import get_file
from artemis.general.ezprofile import EZProfiler, profile_context, get_profile_contexts
from artemis.general.checkpoint_counter import do_every
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from attr import attrs, attrib
from more_itertools import first

from vae_celebA.dfc_vae import encoder, generator
from vae_celebA.image_utils.face_aligner_2 import FaceAligner2, FaceAligner3
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.peters_extensions.cv_image_panels import EasyWindow, Row, cvkey_to_key, Keys, cv_window_input, BGRImageArray, Col
from vae_celebA.peters_extensions.fullscreen_display import show_fullscreen, show_fullscreen_v1
from vae_celebA.peters_extensions.hmc_sampler import hmc_leapfrog_step

LatentArray = 'Array[z_dim:float]'


def demo_decoder(n_steps=1000, step_size=0.1, momentum_refreshment=0.1):
    z_dim = 100
    c_dim = 3
    batch_size = 1
    output_size = 64

    # Setup HMC
    x = np.random.randn(1, z_dim)
    v = np.random.randn(1, z_dim) * 0
    var_x = tf.placeholder(tf.float32, shape=(1, z_dim))
    var_v = tf.placeholder(tf.float32, shape=(1, z_dim))
    x_new, v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum(z ** 2, axis=1), x=var_x, v=var_v, step_size=step_size, momentum_refreshment=momentum_refreshment)

    # Setup Generator
    z_p = tf.zeros([1, z_dim], tf.float32)
    gen0, gen0_logits = generator(z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False)  # reconstruction
    full_img = tf.image.resize_images(gen0.outputs, [224, 224])

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
    v_new = v * momentum - step_size / 2. * gx
    x_new = x + v_new
    return x_new, v_new


def multiply_gaussians(means, variances, axis=0, keepdims=True):
    inverse_vars = 1. / variances
    new_var = 1. / tf.reduce_sum(inverse_vars, axis=axis, keepdims=keepdims)
    new_mean = new_var * tf.reduce_sum(inverse_vars * means, axis=axis, keepdims=keepdims)
    return new_mean, new_var


def crop_by_fraction(im, vcrop, hcrop):
    return im[int(vcrop[0] * im.shape[0]):int(vcrop[1] * im.shape[0]), int(hcrop[0] * im.shape[1]):int(hcrop[1] * im.shape[1])]


def add_fade_frame(img, frame_width=0.05, p_norm=2.):
    r = (np.sum(np.power(np.meshgrid(np.linspace(-1, 1, img.shape[0]), np.linspace(-1, 1, img.shape[1])), p_norm), axis=0)) ** (1. / p_norm)
    fade_mult = (np.minimum(1, np.maximum(0, (1 - r) / frame_width)))[:, :, None]
    bordered_image = (img * fade_mult).astype(np.uint8)
    return bordered_image


def correct_gamma(img, gamma=3.):
    table = (((np.arange(0, 256) / 255.0) ** (1 / gamma)) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def equalize_brightness(img, clipLimit=3., tileGridSize=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def uint8_to_float(img: 'Array[...: uint8]') -> 'Array[H,W,3: uint8]':
    return img / 127.5 - 1.


def float_to_uint8(img: 'Array[...: float]') -> 'Array[H,W,3: uint8]':
    return ((img + 1.) * 127.5).astype(np.uint8)


class VariationalMirror:

    def __init__(self,
                 z_dim=100,
                 c_dim=3,
                 batch_size=1,
                 output_size=64,
                 momentum_refreshment=0.2,
                 step_size=0.05,
                 v_scale=4.,
                 display_size: Optional[Tuple[int, int]] = None
                 ):

        self.x = np.random.randn(1, z_dim)
        self.v = np.random.randn(1, z_dim) * 0
        self.z_dim = z_dim

        with tf.device("/cpu:0"):
            # g = Namespace()
            self.pl_x = tf.placeholder(tf.float32, shape=(1, z_dim))
            self.pl_v = tf.placeholder(tf.float32, shape=(1, z_dim))

            # var_v = tf.placeholder(tf.float32, shape=(1, z_dim))
            self.z_mean = tf.placeholder(tf.float32, shape=(None, z_dim), name='z_mean')
            self.z_var = tf.placeholder(tf.float32, shape=(None, z_dim), name='z_var')

            # Smooth update
            # self.x_new, self.v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum((z-self.z_mean)**2/self.z_var, axis=1), x=self.pl_x, v=self.pl_v, step_size=step_size, momentum_refreshment=momentum_refreshment, v_scale = v_scale*tf.reduce_mean(self.z_var))
            self.x_new, self.v_new = hmc_leapfrog_step(lambda z: 0.5 * tf.reduce_sum(tf.reduce_sum((z - self.z_mean) ** 2 / self.z_var, axis=1), axis=0, keepdims=True),
                                                       x=self.pl_x, v=self.pl_v,
                                                       step_size=step_size, momentum_refreshment=momentum_refreshment, v_scale=v_scale * tf.reduce_mean(self.z_var))
            # self.x_new, self.v_new = momentum_sgd(lambda z: 0.5 * tf.reduce_sum((z-self.z_mean)**2/self.z_var, axis=1), x=pl_x, v=pl_v, step_size=0.01, momentum=0.9)

            # Random update
            self.z_sample = tf.random_normal(shape=(1, z_dim), mean=self.z_mean, stddev=self.z_var ** .5)

            # Setup Generator
            self.z_p = tf.zeros([1, z_dim], tf.float32)
            self.gen0, self.gen0_logits = generator(self.z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False)  # reconstruction

            if display_size is not None:
                self.full_img = tf.image.resize_images(self.gen0.outputs, display_size)
            else:
                self.full_img = self.gen0.outputs

            # Setup Encoder
            self.input_imgs = tf.placeholder(tf.float32, [None, output_size, output_size, c_dim], name='real_images')
            self.net_out1, self.net_out2, self.qz_mean, self.qz_log_sigma_sq = encoder(self.input_imgs, is_train=True, reuse=False, z_dim=z_dim)
            self.qz_var = tf.exp(self.qz_log_sigma_sq)

            self.qz_mean_prod, self.qz_var_prod = multiply_gaussians(means=self.qz_mean, variances=self.qz_var, axis=0, keepdims=True)

        self.sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(self.sess)

        gen_file_path = get_file('models/dfc-vae3/net_self.npz', url='https://drive.google.com/uc?export=download&id=1YHcctf9l90agJSFFSTiMwjm10WGQO6Lu')
        enc_file_path = get_file('models/dfc-vae3/net_e.npz', url='https://drive.google.com/uc?export=download&id=1vLJVU_BDvpDqTfNhC80C3GZI2IxscYwH')
        # LOCAL = True
        # if LOCAL:
        #     netpath = get_file('models/dfc-vae3', url = )
        # else:
        #     netpath = '/home/petered/remote/vae-celebA/vae_celebA/checkpoint/dfc-vae3'

        gen_params = tl.files.load_npz(*os.path.split(gen_file_path))
        tl.files.assign_params(self.sess, gen_params, self.gen0)

        enc_params = tl.files.load_npz(*os.path.split(enc_file_path))
        tl.files.assign_params(self.sess, [enc_params[i] for i in range(25)], self.net_out1)
        tl.files.assign_params(self.sess, [enc_params[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]], self.net_out2)

    def compute_variational_posterior(self, raw_faces: 'Array[N, H,W,3:uint8]') -> Tuple['Array[N,z_dim:float]', 'Array[N,z_dim:float]']:

        if len(raw_faces) > 0:
            raw_faces = raw_faces[[int(time.time() // 10) % len(raw_faces)]]  # Optional
            faces = uint8_to_float(raw_faces)
            # post_mean, post_var = sess.run([self.qz_mean_prod, self.qz_var_prod], {self.input_imgs: faces})
            post_mean, post_var = self.sess.run([self.qz_mean, self.qz_var], {self.input_imgs: faces})
        else:
            post_mean, post_var = np.zeros([1, self.z_dim]), np.ones([1, self.z_dim])
        return post_mean, post_var

    def sample_latent_points(self, post_mean: 'Array[N,z_dim:float]', post_var: 'Array[N,z_dim:float]', smooth=True) -> 'Array[N,z_dim:float]':

        if smooth:
            self.x, self.v = self.sess.run((self.x_new, self.v_new), {self.pl_x: self.x, self.pl_v: self.v, self.z_mean: post_mean, self.z_var: post_var})
            z_points = self.x
        else:
            z_points = self.sess.run(self.z_sample, {self.z_mean: post_mean, self.z_var: post_var})
        return z_points

    def generate_images(self, z_points: 'Array[N,z_dim:float]') -> 'Array[N, H,W,3:uint8]':

        imgs = self.sess.run(self.full_img, {self.z_p: z_points})
        return float_to_uint8(imgs)

    def generate_single_image(self, z_point: LatentArray) -> BGRImageArray:
        img, = self.generate_images(z_point[None])
        return img


def arg_closest(target: 'Array[..., float]', options=Iterable['Array[..., float]']) -> int:
    arg = None
    closest = float('inf')
    for i, op in enumerate(options):
        dist = ((target.ravel() - op.ravel()) ** 2).sum()
        if dist < closest:
            closest = dist
            arg = i
    assert arg is not None, "No options provided to be closest to or all distances infinite"
    return arg


@attrs
class MultiClassfier:
    categories = attrib(type=Dict[str, Dict[str, 'Array[z_dim:float]']], factory=OrderedDict)

    def update(self, category: str, label: str, z_value: 'Array[z_dim:float]'):
        if category not in self.categories:
            self.categories[category] = {}
        self.categories[category][label] = z_value

    def classify(self, z_value='Array[z_dim:float]') -> Mapping[str, str]:
        output = OrderedDict()
        for cat, subdict in self.categories.items():
            arg = arg_closest(target=z_value, options=subdict.values())
            output[cat] = list(subdict.keys())[arg]
        return output


# @attrs(auto_attribs=True)
# class LatentImagePair:
#     latent_value: LatentArray
#     BGRImageArray: BGRImageArray
#
# class AnalogyValues(Enum):
#     A = 'a'
#     B = 'b'
#     C = 'c'
#     D = 'd'
#
# @attrs(auto_attribs=True)
# class AnalogyMaker:
#     """ Reasons a "A is to B as C is to D" relation
#
#     ie. A - B = C - D
#     """
#     mapping = Mapping[AnalogyValues, LatentArray]
#
#     def set_value(self, variable: AnalogyValues, value: LatentArray):
#
#         self.mapping[variable] = value
#
#     def query_value(self, variable):
#
#
#
#         if variable == AnalogyValues.A:
#             if AnalogyValues.B and AnalogyValues.C and AnalogyValues.D:
#                 return


@attrs
class AnalogyRenderer:
    mapping_func = attrib(type=Callable[[LatentArray], BGRImageArray])
    values = attrib(type=Dict[str, LatentArray], factory=dict)
    window = attrib(type=EasyWindow, factory=lambda: EasyWindow(Col(Row('A', 'B'), Row('C', 'D'))))
    scale = attrib(type=float, default=2)
    title_mapping = attrib(type=Mapping[str, str], default={'A': "A is to", 'B': 'B', 'C': "as C is to", 'D': 'D'})

    def set_value(self, variable: str, value: LatentArray):

        self.values[variable] = value
        if len(self.values) > 3:
            self.values.clear()
            self.values[variable] = value

    def clear(self):
        self.values.clear()

    def get_analogy_vector(self) -> Optional[LatentArray]:
        if 'A' in self.values and 'B' in self.values:
            return self.values['B'] - self.values['A']
        elif 'C' in self.values and 'D' in self.values:
            return self.values['D'] - self.values['C']
        else:
            return None

    def render(self) -> BGRImageArray:

        image = None
        for variable in 'ABCD':
            if variable in self.values:
                image = self.mapping_func(self.values[variable])
                self.window.update(image, name=variable, scale=self.scale, title=self.title_mapping[variable])

        if image is None:
            return None

        print(f'Plotting with {len(self.values)}')
        if len(self.values) == 3:
            print('Computing missing')
            missing = first(v for v in 'ABCD' if v not in self.values)

            final_z = \
                self.values['B'] + self.values['C'] - self.values['D'] if missing == 'A' else \
                    self.values['A'] - self.values['C'] + self.values['D'] if missing == 'B' else \
                        self.values['A'] - self.values['B'] + self.values['D'] if missing == 'C' else \
                            -self.values['A'] + self.values['B'] + self.values['C']

            image = self.mapping_func(final_z)
            image[[0, -1]] = (0, 255, 0)  # Border
            image[:, [0, -1]] = (0, 255, 0)  # Border
            self.window.update(image=image, name=missing, scale=self.scale, title=self.title_mapping[missing])
        else:
            default_image = image * 0
            for variable in 'ABCD':
                if variable not in self.values:
                    self.window.update(default_image, name=variable, scale=self.scale, title=self.title_mapping[variable])

        return self.window.render()


ShowFunc = Callable[[BGRImageArray], str]


@attrs(auto_attribs=True)
class OpenCVShowFunc(ShowFunc):
    window_name: str = 'window'

    def __call__(self, img: BGRImageArray) -> str:
        cv2.imshow(self.window_name, img)
        return cvkey_to_key(cv2.waitKey(1))


@attrs(auto_attribs=True)
class FullScreenShowFunc(ShowFunc):
    display_sizes: Sequence[Tuple[int, int]] = [(1440, 900), (1920, 1080)]
    display_number: int = 0

    def __call__(self, img: BGRImageArray) -> str:
        return cvkey_to_key(show_fullscreen(image=img, background_colour=(0, 0, 0), display_sizes=self.display_sizes, display_number=self.display_number))


class WindowNames:
    MIRROR = 'Mirror'
    DETECTIONS = 'Detections'
    ANALOGIES = 'Analogies'


def demo_var_mirror(
        n_steps=None,
        video_size=(320, 240),
        smooth=True,
        show_display_plot=False,
        show_camera_window=False,
        opposite=False,
        camera_device_no=0,
        crop_frac=None,
        do_brightness_equalization=True,
        show_func: ShowFunc = OpenCVShowFunc(),
        apply_analogy=False,
):
    """

    """
    vm = VariationalMirror()

    # face_detector = FaceAligner2(
    #     desiredLeftEye=[0.35954122, 0.51964207],
    #     desiredRightEye=[0.62294991, 0.52083333],
    #     desiredFaceWidth=64,
    #     desiredFaceHeight=64,
    # )

    face_detector = FaceAligner3(
        # desired_left_eye_xy=(.333, .5),
        desired_left_eye_xy=(.36, .515),
        desired_right_eye_xy=(1-.38, .515),
        # desired_right_eye_xy=(.5, .5),
        desired_width=64,
    )

    cam = VideoCamera(size=video_size, device=camera_device_no, hflip=True, mode='rgb')

    window = EasyWindow(Row(
        Col(WindowNames.DETECTIONS, WindowNames.ANALOGIES),
        WindowNames.MIRROR
    ), panel_scales={WindowNames.MIRROR: 10})

    mc = MultiClassfier()

    # analogy_values: Dict[str, Tuple[LatentArray]] = dict()

    analogy_renderer = AnalogyRenderer(mapping_func=lambda z: vm.generate_single_image(z)[:, :, ::-1])

    show_input_in_mirror = False
    # Run
    # im = np.zeros((64, 64, 3))
    for t in range(n_steps) if n_steps is not None else itertools.count(0):

        with profile_context('total'):
            rgb_im = cam.read()

            if rgb_im is None:
                print('No Camera Image')
                time.sleep(0.1)
                continue

            if crop_frac is not None:
                rgb_im = crop_by_fraction(rgb_im, *crop_frac)

            if do_brightness_equalization:
                rgb_im = correct_gamma(rgb_im)

            with profile_context('detection'):
                landmarks, raw_faces = face_detector(rgb_im)

            with profile_context('inference'):
                if len(raw_faces) > 0:
                    raw_faces = raw_faces[[int(time.time() // 10) % len(raw_faces)]]  # Optional

                post_mean, post_var = vm.compute_variational_posterior(raw_faces)

            if opposite:
                post_mean = -post_mean

            z_points = vm.sample_latent_points(post_mean=post_mean, post_var=post_var, smooth=smooth)

            with profile_context('generation'):
                z_shift = analogy_renderer.get_analogy_vector() if apply_analogy else None
                z_points_maybe_shifted = z_points if z_shift is None else z_points + z_shift
                generated_image = vm.generate_images(z_points_maybe_shifted)[0, :, :, ::-1]

            with profile_context('rendering'):
                if show_display_plot:
                    # classification = mc.classify(post_mean)
                    # classification_string = ', '.join(f"{k}:{v}" for k, v in classification.items())
                    if show_input_in_mirror and len(raw_faces) > 0:  # Just for debugging
                        window.update(image=cv2.resize(raw_faces[0, :, :, ::-1], dsize=(generated_image.shape[1], generated_image.shape[0])), name=WindowNames.MIRROR,
                                      title='input')
                    else:
                        title = WindowNames.MIRROR
                        if opposite:
                            title += ', Opposite Mode'
                        if apply_analogy:
                            title += ', Applying Analogy A->B'
                        window.update(image=generated_image, name=WindowNames.MIRROR, title=title)

                if show_camera_window:
                    display_img = rgb_im[..., ::-1].copy()

                    for landmark, face in zip(landmarks, raw_faces):
                        cv2.circle(display_img, tuple(landmark.left_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
                        cv2.circle(display_img, tuple(landmark.right_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
                        display_img[-face.shape[0]:, -face.shape[1]:, ::-1] = face

                    window.update(image=display_img, name=WindowNames.DETECTIONS)

                full_image = window.render()
                key = show_func(full_image)
                if key == Keys.C:
                    result = cv_window_input(prompt='Enter "<property>:<value>".  For example: "gender:female"')
                    if result:
                        if '=' in result:
                            category, label = result.split('=', 1)
                            mc.update(category=category, label=label, z_value=post_mean)
                        else:
                            print(f'Invalid input {result}')
                elif key == Keys.S:
                    smooth = not smooth
                elif key == Keys.O:
                    opposite = not opposite
                elif key == Keys.A:
                    apply_analogy = not apply_analogy
                elif key == Keys.I and len(raw_faces) > 0:
                    show_input_in_mirror = not show_input_in_mirror
                elif key in (Keys.n1, Keys.n2, Keys.n3, Keys.n4, Keys.n0):
                    if key == Keys.n0:
                        analogy_renderer.clear()
                    else:
                        key_to_name_mapping = {Keys.n1: 'A', Keys.n2: 'B', Keys.n3: 'C', Keys.n4: 'D'}
                        analogy_renderer.set_value(key_to_name_mapping[key], value=post_mean[0])
                    analogy_image = analogy_renderer.render()
                    window.update(image=analogy_image, name=WindowNames.ANALOGIES)
                elif key == Keys.ESC:
                    break

        if do_every('5s'):
            profile = get_profile_contexts(['total', 'detection', 'generation', 'inference', 'rendering'], fill_empty_with_zero=True)
            print(
                f'Mean Times:: Total: {profile["total"][1] / profile["total"][0]:.3g}, Detection: {profile["detection"][1] / profile["detection"][0]:.3g}, '
                f'Inference: {profile["inference"][1] / profile["inference"][0]:.3g}, Generation: {profile["generation"][1] / profile["generation"][0]:.3g}, '
                f'Rendering: {profile["rendering"][1] / profile["rendering"][0]:.3g}'
            )


if __name__ == '__main__':

    OUTSIDE = False
    # display_sizes = [(1440, 900), (1920, 1080)]
    display_sizes = [(1920, 1080), (1920, 1080)]

    if OUTSIDE:
        crop_frac = [(.3, .7), (0, 1)]
        video_size = (640, 480)
    else:
        crop_frac = None
        video_size = (320, 240)

    demo_var_mirror(
        smooth=True,
        opposite=False,
        show_display_plot=True,
        show_camera_window=True,
        camera_device_no=0,
        video_size=video_size,
        crop_frac=crop_frac,
        do_brightness_equalization=False,
        # show_func=FullScreenShowFunc(display_sizes=[(1920, 1080), (1920, 1080)])
    )
