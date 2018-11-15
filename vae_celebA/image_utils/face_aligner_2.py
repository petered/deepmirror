

# import the necessary packages
import face_recognition
import numpy as np
import cv2
import dlib

from artemis.general.ezprofile import EZProfiler

"""
Taken from the website of Adrian Rosebrock
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
"""

# import the necessary packages
from collections import OrderedDict
from artemis.fileman.file_getter import get_file
import numpy as np
import cv2


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


class FaceAligner2:

    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredRightEye=None, desiredFaceWidth=256, desiredFaceHeight=None, border_mode = cv2.BORDER_REPLICATE, model='large'):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        # self.detector=detector
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        self.desiredRightEye = desiredRightEye
        self.border_mode = border_mode
        self.model = model

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def __call__(self, im):

        with EZProfiler('landmark_det'):
            landmarks_per_face = face_recognition.face_landmarks(im, model=self.model)

        landmarks_per_face = sorted(landmarks_per_face, key = lambda x: x['left_eye'][0][0])


        ims = []
        for landmarks in landmarks_per_face:
            ims.append(self.align(im, landmarks))
        landmarks_per_face = [{k: np.array(v).copy() for k, v in landmarks.items()} for landmarks in landmarks_per_face]
        return landmarks_per_face, np.array(ims)


    def align(self, image, landmarks):
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
        leftEyeCenter = np.mean(landmarks['left_eye'], axis=0)
        rightEyeCenter = np.mean(landmarks['right_eye'], axis=0)

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
