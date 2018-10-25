import face_recognition
from future.moves import itertools

from artemis.general.global_rates import measure_global_rate
from vae_celebA.image_utils.face_aligner import FaceAligner
from vae_celebA.image_utils.face_aligner_2 import FaceAligner2
from vae_celebA.utils import get_image

__author__ = 'peter'
import time
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.fileman.smart_io import smart_load_image
from vae_celebA.image_utils.video_camera import VideoCamera
from artemis.fileman.file_getter import get_artemis_data_path
from artemis.general.global_rates import measure_global_rate
import itertools
from artemis.general.ezprofile import EZProfiler


def demo_aligned_face_detection():

    cam = VideoCamera(size=(320, 180))

    sample_pic = smart_load_image(get_artemis_data_path('data/celeba/img_align_celeba/000001.jpg'))

    face_detector = FaceAligner.build_default(
        desiredLeftEye = (0.39, 0.51),
        desiredFaceWidth=sample_pic.shape[1],
        desiredFaceHeight=sample_pic.shape[0],
    )

    for i in itertools.count(0):
        sample_pic = smart_load_image(get_artemis_data_path(f'data/celeba/img_align_celeba/{i+1:06d}.jpg'))
        with hold_dbplots():
            dbplot(sample_pic, 'sample_pic')
            bgr_im = cam.read()
            if bgr_im is not None:
                rgb_im = bgr_im[..., ::-1]
                faces = face_detector(rgb_im)
                dbplot(rgb_im, 'You')
                dbplot(faces, 'Detected Faces')
                # dbplot([(f[:, :, ::-1].astype(np.float) + .5*sample_pic.astype(np.float).mean(axis=-1, keepdims=True))/2 for f in faces], 'Merge Faces')
                dbplot(sample_pic if len(faces)==0 else faces*0+sample_pic if i%2==0 else faces, 'Swap Faces')
            else:
                print('Camera Framed Dropped!')
                time.sleep(0.1)


def define_eye_positions(n_pics=10):

    left = []
    right = []
    for i in range(n_pics):
        sample_pic = smart_load_image(get_artemis_data_path(f'data/celeba/img_align_celeba/{i+1:06d}.jpg'))

        im = get_image(get_artemis_data_path(f'data/celeba/img_align_celeba/{i+1:06d}.jpg'), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)

        landmarks = face_recognition.face_landmarks(((im+1)*127.5).astype(np.uint8), model='large')
        dbplot(im, 'im')
        if len(landmarks)==0:
            print(f'No face found in image {i}')
        else:
            lmark = landmarks[0]
            left_eye_position = np.mean(lmark['left_eye'], axis=0)/(im.shape[1], im.shape[0])
            right_eye_position = np.mean(lmark['right_eye'], axis=0)/(im.shape[1], im.shape[0])
            print(f'Left-Eye_position: {left_eye_position}')
            print(f'Right-Eye_position: {right_eye_position}')
            left.append(left_eye_position)
            right.append(right_eye_position)

    print(f'Mean Position: Left: {np.mean(left, axis=0)}±{np.std(left, axis=0)}, Right: {np.mean(right, axis=0)}+{np.std(left, axis=0)}')


def demo_aligned_face_detection_2():

    cam = VideoCamera(size=(320, 180))
    # cam = VideoCamera(size=(640, 480))
    i=0
    sample_pic = get_image(get_artemis_data_path(f'data/celeba/img_align_celeba/{i//10+1:06d}.jpg'), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)

    face_detector = FaceAligner2(
        # desiredLeftEye = (0.39, 0.51),
        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        desiredFaceWidth=sample_pic.shape[1],
        desiredFaceHeight=sample_pic.shape[0],
    )

    for i in itertools.count(0):
        sample_pic = get_image(get_artemis_data_path(f'data/celeba/img_align_celeba/{i//10+1:06d}.jpg'), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)


        with hold_dbplots():
            dbplot(sample_pic, 'sample_pic')
            bgr_im = cam.read()
            if bgr_im is not None:
                rgb_im = bgr_im[..., ::-1]

                # with EZProfiler('loc detection'):
                #     face_locations = face_recognition.face_locations(rgb_im)
                #
                # with EZProfiler('landmark detection'):
                #     landmarks = face_recognition.face_landmarks(rgb_im)

                with EZProfiler('alignment'):
                    faces = face_detector(rgb_im)
                # print(f'Mean Rate: {measure_global_rate("XX"):.3g}iter/s')

                # faces = face_detector(rgb_im)
                dbplot(rgb_im, 'You')

                if len(faces)>0:
                    dbplot(faces, 'Detected Faces')
                # dbplot([(f[:, :, ::-1].astype(np.float) + .5*sample_pic.astype(np.float).mean(axis=-1, keepdims=True))/2 for f in faces], 'Merge Faces')
                frac = np.sin(i/5.)*.5 + .5
                dbplot(sample_pic if len(faces)==0 else faces*(1-frac)+sample_pic*frac, 'Swap Faces')
            else:
                print('Camera Framed Dropped!')
                time.sleep(0.1)


if __name__ == '__main__':
    # define_eye_positions(n_pics=100)
    demo_aligned_face_detection_2()
