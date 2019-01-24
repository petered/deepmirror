from vae_celebA.image_utils.face_aligner_2 import FaceAligner2
from vae_celebA.image_utils.video_camera import VideoCamera
from vae_celebA.peters_extensions.attention_window import AttentionWindow
import cv2

if __name__ == '__main__':

    FAST = False
    # windower = AttentionWindow(window_size=(200, 200), decay_rate=0.5, downsample_to=(80, 60), boredom_decay=0.0)
    windower = AttentionWindow.from_default_settings('faces')

    face_detector = FaceAligner2()

    if FAST:
        for im in VideoCamera(size=(640, 480), device=0, mode='bgr').iterator():
            crop = windower(im)

            landmarks, faces = face_detector(crop)


            display_im = im.copy()
            display_im[-crop.shape[0]:, -crop.shape[1]:] = crop
            cv2.imshow('win', display_im)
            cv2.waitKey(1)
    else:
        for im in VideoCamera(size=(640, 480), device=0, mode='rgb').iterator():
            attention, crop, bbox = windower.get_attention_window_and_crop(im)
            with hold_dbplots():
                dbplot(im, 'image')
                dbplot(bbox, 'bbox', axis='image', plot_type='bbox')
                dbplot(attention, 'attention')
                dbplot(crop, 'crop')
