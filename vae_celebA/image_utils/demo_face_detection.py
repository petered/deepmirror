__author__ = 'peter'
import time
from artemis.plotting.db_plotting import dbplot
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.image_utils.face_detection import FaceDetectorExtractor, get_face_detector_version


def demo_face_detection():

    cam = VideoCamera(size=(320, 180))

    face_detector = get_face_detector_version('video')

    while True:
        bgr_im = cam.read()
        if bgr_im is not None:
            faces = face_detector(bgr_im)
            dbplot(bgr_im[:, ::-1, ::-1], 'You', draw_now=False)
            dbplot([f[:, :, ::-1] for f in faces], 'Detected Faces')
        else:
            print('Camera Framed Dropped!')
            time.sleep(0.1)


if __name__ == '__main__':
    demo_face_detection()
