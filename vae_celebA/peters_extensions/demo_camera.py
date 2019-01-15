

if __name__ == '__main__':
    import itertools
    from vae_celebA.image_utils.video_camera import VideoCamera
    import cv2

    cam = VideoCamera(size=(640, 480), device=0, mode='bgr')

    for t in itertools.count(0):
        im = cam.read()
        if im is not None:
            cv2.imshow('im', im)
            cv2.waitKey(1)
        else:
            print('No Camera Image')
