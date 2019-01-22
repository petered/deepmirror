import face_recognition
from functools import partial

from artemis.general.async import iter_latest_asynchonously
from vae_celebA.image_utils.face_aligner import FaceAligner
from vae_celebA.image_utils.face_aligner_2 import FaceAligner2, face_aligning_iterator, display_face_aligner
from vae_celebA.utils import get_image

__author__ = 'peter'
import time
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.fileman.smart_io import smart_load_image
from vae_celebA.image_utils.video_camera import VideoCamera
from artemis.fileman.file_getter import get_artemis_data_path
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
        sample_pic = smart_load_image(get_artemis_data_path('data/celeba/img_align_celeba/{:06d}.jpg'.format(i+1)))
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
        sample_pic = smart_load_image(get_artemis_data_path('data/celeba/img_align_celeba/{:06d}.jpg'.format(i+1)))

        im = get_image(get_artemis_data_path('data/celeba/img_align_celeba/{i+1:06d}.jpg'), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)

        landmarks = face_recognition.face_landmarks(((im+1)*127.5).astype(np.uint8), model='large')
        dbplot(im, 'im')
        if len(landmarks)==0:
            print('No face found in image {}'.format(i))
        else:
            lmark = landmarks[0]
            left_eye_position = np.mean(lmark['left_eye'], axis=0)/(im.shape[1], im.shape[0])
            right_eye_position = np.mean(lmark['right_eye'], axis=0)/(im.shape[1], im.shape[0])
            print('Left-Eye_position: {}'.format(left_eye_position))
            print('Right-Eye_position: {}'.format(right_eye_position))
            left.append(left_eye_position)
            right.append(right_eye_position)

    print('Mean Position: Left: {}±{}, Right: {}+{}'.format(np.mean(left, axis=0), np.std(left, axis=0), np.mean(right, axis=0), np.std(left, axis=0)))


def demo_aligned_face_detection_2(camera_device_no = 0, camera_size=(320, 240), model='large'):

    cam = VideoCamera(size=camera_size, device=camera_device_no)
    # cam = VideoCamera(size=(640, 480))
    i=0
    sample_pic = get_image(get_artemis_data_path('data/celeba/img_align_celeba/{:06d}.jpg'.format(i//10+1)), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)

    face_detector = FaceAligner2(
        # desiredLeftEye = (0.39, 0.51),
        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        desiredFaceWidth=sample_pic.shape[1],
        desiredFaceHeight=sample_pic.shape[0],
        model = model,
    )

    for i in itertools.count(0):
        sample_pic = get_image(get_artemis_data_path('data/celeba/img_align_celeba/{:06d}.jpg'.format(i//10+1)), image_size=148, is_crop=True, resize_w=64, is_grayscale = 0)

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
                    landmarks, faces = face_detector(rgb_im)
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



def demo_aligned_face_detection_simple(camera_device_no = 0, camera_size=(320, 240), face_size=(64, 64), model='large'):

    cam = VideoCamera(size=camera_size, device=camera_device_no)
    # cam = VideoCamera(size=(640, 480))
    i=0

    face_width, face_height = face_size
    face_detector = FaceAligner2(
        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        desiredFaceWidth=face_width,
        desiredFaceHeight=face_height,
        model = model,
    )

    for i in itertools.count(0):
        with hold_dbplots():
            bgr_im = cam.read()
            if bgr_im is not None:
                rgb_im = bgr_im[..., ::-1]

                with EZProfiler('alignment'):
                    landmarks, faces = face_detector(rgb_im)

                dbplot(rgb_im, 'You')

                dbplot(faces if len(faces)>0 else np.zeros((1, )+faces.shape[1:], dtype=np.uint8), 'Detected Faces')
            else:
                print('Camera Framed Dropped!')
                time.sleep(0.1)


def demo_face_aligner_iterator(async=False, size=(320, 240)):

    face_aligner=FaceAligner2(
        desiredLeftEye = [0.35954122, 0.51964207],
        desiredRightEye = [0.62294991, 0.52083333],
        desiredFaceWidth=64,
        desiredFaceHeight=64,
        model = 'large',
        )
    camera = VideoCamera(size=size, mode='rgb')

    gen_func = partial(face_aligning_iterator, face_aligner=face_aligner, camera=camera)

    if async:
        iterator = iter_latest_asynchonously(
            gen_func = gen_func, empty_value=(None, None, None),
            use_forkserver=True,
            uninitialized_wait=0.1
        )
    else:
        iterator = gen_func()

    for t, (img, landmarks, faces) in enumerate(iterator):
        if img is None:
            print('No Camera Image')
            time.sleep(0.1)
            continue

        display_face_aligner(img, landmarks, faces, text = 't={}: {} Faces'.format(t, len(landmarks)))

        # dbplot(img, 'You', cornertext=f't={t}')
        # dbplot(faces if len(faces)>0 else np.zeros((1, )+faces.shape[1:], dtype=np.uint8), 'Detected Faces')



if __name__ == '__main__':
    # define_eye_positions(n_pics=100)
    # demo_aligned_face_detection_2(camera_device_no=0, camera_size = (640, 480), model='large')
    # demo_aligned_face_detection_simple(camera_device_no=0, camera_size = (640, 480), model='large')
    # demo_face_aligner_iterator(size=(800, 600))
    demo_face_aligner_iterator(size=(1024, 768))