

if __name__ == '__main__':
    import itertools
    from vae_celebA.image_utils.video_camera import VideoCamera
    from artemis.plotting.db_plotting import dbplot, hold_dbplots
    import cv2

    cam = VideoCamera(size=(640, 480), device=0, mode='rgb')

    for t in itertools.count(0):
        im = cam.read()
        if im is not None:
            dbplot(im, 'im')
        else:
            print('No Camera Image')
