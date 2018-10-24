from future.moves import itertools

from vae_celebA.image_utils.face_aligner import FaceAligner

__author__ = 'peter'
import time
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.fileman.smart_io import smart_load_image
from vae_celebA.image_utils.video_camera import VideoCamera
from artemis.fileman.file_getter import get_artemis_data_path
import itertools


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
                dbplot(faces*0+sample_pic if i%2==0 else faces, 'Swap Faces')
            else:
                print('Camera Framed Dropped!')
                time.sleep(0.1)


if __name__ == '__main__':
    demo_aligned_face_detection()
