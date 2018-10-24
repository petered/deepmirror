import time
from artemis.plotting.db_plotting import dbplot
from vae_celebA.image_utils.video_camera import VideoCamera


def demo_camera():

    cam = VideoCamera(size=(320, 180))
    while True:
        bgr_im = cam.read()
        if bgr_im is not None:
            dbplot(bgr_im[:, ::-1, ::-1], 'You')
        else:
            print('Camera Framed Dropped!')
            time.sleep(0.1)


if __name__ == '__main__':
    demo_camera()
