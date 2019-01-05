

from collections import OrderedDict
from collections import namedtuple

import cv2
import face_recognition
import numpy as np
from typing import Tuple, Sequence

from artemis.general.checkpoint_counter import do_every
from artemis.general.ezprofile import profile_context, get_profile_contexts_string
from vae_celebA.image_utils.video_camera import VideoCamera

"""
Extended from stuff on the website of Adrian Rosebrock
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
"""


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


# class FaceLandmarks(NamedTuple):
#     """
#     Each is an (n_points, 2) array of (x, y) locations.
#     """
#     nose_tip: np.ndarray
#     left_eye: np.ndarray
#     right_eye: np.ndarray
#     chin: np.ndarray = None
#     left_eyebrow: np.ndarray = None
#     right_eyebrow: np.ndarray = None
#     nose_bridge: np.ndarray = None
#     top_lip: np.ndarray = None
#     bottom_lip: np.ndarray = None



# FaceLandmarks = namedtuple('FaceLandmarks', ['nose_tip', 'left_eye', 'right_eye', 'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'top_lip', 'bottom_lip'])
FaceLandmarks = namedtuple('FaceLandmarks', ['nose_tip', 'left_eye', 'right_eye'])


class FaceAligner2:

    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredRightEye=None, desiredFaceWidth=256, desiredFaceHeight=None, border_mode = cv2.BORDER_REPLICATE, model='small'):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        # self.detector=detector
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        self.desiredRightEye = desiredRightEye
        self.border_mode = border_mode
        self.model='small'

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def __call__(self, im: np.ndarray) -> Tuple[Sequence[FaceLandmarks], np.ndarray]:

        landmark_dicts = face_recognition.face_landmarks(im, model=self.model)
        landmarks_per_face = [FaceLandmarks(**{k: np.array(v) for k, v in d.items()}) for d in landmark_dicts]

        landmarks_per_face = sorted(landmarks_per_face, key = lambda x: x.left_eye[0][0])
        ims = []
        for landmarks in landmarks_per_face:
            ims.append(self.align(im, landmarks))
        return landmarks_per_face, np.array(ims) if len(ims)>0 else np.zeros((0, self.desiredFaceHeight, self.desiredFaceWidth, 3))

    def align(self, image, landmarks: FaceLandmarks):
        #
        # # convert the landmark (x, y)-coordinates to a NumPy array
        # shape = self.predictor(gray, rect)
        # shape = shape_to_np(shape)
        #
        # # extract the left and right eye (x, y)-coordinates
        # (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        # leftEyePts = shape[lStart:lEnd]
        # rightEyePts = shape[rStart:rEnd]
        #
        # # compute the center of mass for each eye
        # leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        # rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # leftEyeCenter = np.mean(landmarks['left_eye'], axis=0).astype("int")
        # rightEyeCenter = np.mean(landmarks['right_eye'], axis=0).astype("int")
        leftEyeCenter = np.mean(landmarks.left_eye, axis=0)
        rightEyeCenter = np.mean(landmarks.right_eye, axis=0)

        # leftEyeCenter[1] = image.shape[0] - leftEyeCenter[1]
        # rightEyeCenter[1] = image.shape[0] - rightEyeCenter[1]

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0] if self.desiredRightEye is None else self.desiredRightEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=self.border_mode)

        # return the aligned face
        return output




def crop_by_fraction(im, vcrop, hcrop):
    return im[int(vcrop[0]*im.shape[0]):int(vcrop[1]*im.shape[0]), int(hcrop[0]*im.shape[1]):int(hcrop[1]*im.shape[1])]


def add_fade_frame(img, frame_width=0.05, p_norm=2.):

    r = (np.sum(np.power(np.meshgrid(np.linspace(-1, 1, img.shape[0]), np.linspace(-1, 1, img.shape[1])), p_norm), axis=0))**(1./p_norm)
    fade_mult = (np.minimum(1, np.maximum(0, (1-r)/frame_width)))[:, :, None]
    bordered_image = (img*fade_mult).astype(np.uint8)
    return bordered_image


def correct_gamma(img, gamma = 3.):
    table = (((np.arange(0, 256)/255.0)**(1/gamma))*255).astype(np.uint8)
    return cv2.LUT(img, table)


def equalize_brightness(img, clipLimit=3., tileGridSize=(8, 8)):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def face_aligning_iterator(face_aligner: FaceAligner2, camera: VideoCamera, image_preprocessor=None):

    for im in camera.iterator():
        if im is None:
            yield None, None, None
        else:
            if image_preprocessor is not None:
                with profile_context('preprocessing'):
                    im = image_preprocessor(im)
            with profile_context('face_detection'):
                landmarks, faces = face_aligner(im)

            if do_every('5s'):
                print(get_profile_contexts_string(fill_empty_with_zero=True))

            yield im, landmarks, faces


def display_face_aligner(rgb_im, landmarks, faces, text=None):
    display_img = rgb_im[..., ::-1].copy()

    for i, (landmark, face) in enumerate(zip(landmarks, faces)):
        cv2.circle(display_img, tuple(landmark.left_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
        cv2.circle(display_img, tuple(landmark.right_eye.mean(axis=0).astype(int)), radius=5, thickness=2, color=(0, 0, 255))
        display_img[-face.shape[0]:, -face.shape[1]*(i+1):display_img.shape[1]-face.shape[1]*i, ::-1] = face
    if text is not None:
        cv2.putText(display_img, text, (20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 0))
    cv2.imshow('camera', display_img)
    cv2.waitKey(1)

# def faces_generator(face_aligner: FaceAligner2, lock, namespace):
#
#     while True:
#         with lock:
#             image = namespace.image
#         landmarks, faces = face_aligner(image)
#         with lock:
#             namespace.imfaces = landmarks, faces

#
# def async_face_aligning_iterator(face_aligner: FaceAligner2, camera: VideoCamera, image_preprocessor):
#
#     m = Manager()
#     namespace = m.Namespace()
#
#     lock = Lock()
#
#
#
#     for im in camera.iterator():
