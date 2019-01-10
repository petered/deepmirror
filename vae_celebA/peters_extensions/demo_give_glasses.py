import itertools

from vae_celebA.image_utils.face_aligner_2 import FaceAligner2
from vae_celebA.image_utils.video_camera import VideoCamera
from artemis.plotting.db_plotting import dbplot, hold_dbplots
import cv2

intify = lambda x: tuple(x.astype(int))


def add_glasses_(img, landmark, color=(0, 0, 0)):

    left_ctr = landmark.left_eye.mean(axis=0)
    right_ctr = landmark.right_eye.mean(axis=0)



    rad = int(((right_ctr-left_ctr)**2).sum()**.5 / 3.)
    thickness = int(rad/4)

    cv2.circle(img, intify(left_ctr), radius=rad, thickness=thickness, color=color)
    cv2.circle(img, intify(right_ctr), radius=rad, thickness=thickness, color=color)

    cv2.line(img, intify(left_ctr+(-rad, -rad)), intify(right_ctr+(rad, -rad)), color=color, thickness=thickness)  # rim

    cv2.line(img, intify(left_ctr+(-rad, -rad)), intify(landmark.chin[0] + (0, -rad/2)), color=color, thickness=thickness)  # right earpiece
    cv2.line(img, intify(right_ctr+(+rad, -rad)), intify(landmark.chin[-1] + (0, -rad/2)), color=color, thickness=thickness)  # left earpiece
    cv2.line(img, intify(left_ctr+(-rad, -rad)), intify(left_ctr+(-rad, 0)), color=color, thickness=thickness)  # right earpiece
    cv2.line(img, intify(right_ctr+(+rad, -rad)),  intify(right_ctr+(+rad, 0)), color=color, thickness=thickness)  # left earpiece

    cv2.ellipse(img, center = intify((left_ctr+right_ctr)/2), axes=(rad*2//3, rad*2//3), angle=0, startAngle=-40, endAngle=-140, color=color, thickness=thickness)


def add_lipstick_(img, landmark, color=(0, 0, 128)):
    cv2.fillPoly(img, pts=[landmark.top_lip.astype(int)], color=color, )
    cv2.fillPoly(img, pts=[landmark.bottom_lip.astype(int)], color=color, )

def enhance_eyebrows_(img, landmark, color=(0, 20, 20)):


    cv2.polylines(img, pts=[landmark.left_eyebrow.astype(int), landmark.right_eyebrow.astype(int)], color=color, thickness=5, isClosed=False)
    # cv2.fillPoly(img, pts=[landmark.right_eyebrow.astype(int)], color=color, )


def demo_give_glasses():

    cam = VideoCamera(size=(640, 480), device=0, mode='bgr', hflip=True)
    face_det = FaceAligner2(model='large')

    for im in cam.iterator():

        im = im.copy()
        landmarks, faces = face_det(im)

        for landmark in landmarks:
            # enhance_eyebrows_(im, landmark)
            add_glasses_(im, landmark)
            add_lipstick_(im, landmark)

        cv2.imshow('image', im)
        cv2.waitKey(1)


if __name__ == '__main__':
    demo_give_glasses()
