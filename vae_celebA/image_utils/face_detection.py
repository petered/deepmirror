import cv2
from artemis.fileman.config_files import get_config_value
import numpy as np
import os

__author__ = 'peter'


class FaceBoundingBoxExtractor():

    def __init__(self, template = 'haarcascade_frontalface_default', min_relative_size = 0.1, face_detector_args = {}):
        """
        :param template: One of: {'haarcascade_frontalface_default', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt_tree'}
            Listed in decreasing order of recall rate (increasing order of precision)
        :param eye_filter:
        :param crop_extension:
        :param face_detector_args:
        :param min_relative_size: Minimum proportion of picture size that face can take.
        :return:
        """
        self._memo_hashable = locals()
        del self._memo_hashable['self']
        haarcascade_folder = get_config_value('.quvacoderc', section='cv', option='haarcascades', default_generator=get_haarcascade_folder, write_default=True)
        # TODO: Better way of locating
        self.face_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_folder, "%s.xml" % (template, )))
        self.face_detector_args = face_detector_args
        if min_relative_size is not None:
            assert 'minSize' not in face_detector_args, "You can't specify both the face detector argument minSize and min_relative_size"
        self.min_relative_size = min_relative_size

    def __call__(self, im):
        """
        :param im: A (size_y, size_x, 3) RGB or RGB image (it's cast to greyscale anyway)
        :return: A list of (crop_size_y, crop_size_x, 3) arrays containing cropped faces
        """
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim==3 else im
        # gray_im = cv2.equalizeHist(gray_im)

        if self.min_relative_size is not None:
            face_bounding_boxes = self.face_cascade.detectMultiScale(gray_im, minSize = (int(self.min_relative_size*im.shape[0]), int(self.min_relative_size*im.shape[1])), **self.face_detector_args)
        else:
            face_bounding_boxes = self.face_cascade.detectMultiScale(gray_im, **self.face_detector_args)
        return face_bounding_boxes

    def memo_hashable(self):
        return self._memo_hashable


class SingleFaceBoundingBox(object):

    def __init__(self, memory = 0.9):
        assert 0 <= memory < 1
        self._current_bb = np.array([0, 0, 1, 1])
        self._old_score = 0
        self.memory = memory

    def __call__(self, bounding_boxes):
        if len(bounding_boxes)==0:
            return [self._current_bb]
        #     return []
        self._old_score *= self.memory
        bounding_boxes = np.array(bounding_boxes)
        new_bb_areas = bounding_boxes[:, 2]*bounding_boxes[:, 3]
        overlaps = bb_overlap(self._current_bb, bounding_boxes)
        new_scores = self._old_score * overlaps + new_bb_areas
        if np.max(new_scores) > self._old_score:
            selected_bbox = np.argmax(new_scores)
            out_box = bounding_boxes[selected_bbox]
            self._old_score = new_scores[selected_bbox]
            self._current_bb = out_box
        else:
            out_box = self._current_bb

        return [out_box]



class MultiFaceBoundingBox(object):

    def __init__(self, memory = 0.9):
        assert 0 <= memory < 1
        self._current_boxes = np.zeros((0, 4))
        self._old_score = np.zeros((0, 0))
        self.memory = memory

    def __call__(self, proposal_boxes):
        self._old_score *= self.memory

        proposal_areas = proposal_boxes[:, 2] * proposal_boxes[:, 3]

        overlaps = np.array([bb_overlap(bb, proposal_boxes) for bb in self._current_boxes])  # (n_old, n_new)

        scores = self._old_score[:, None] * overlaps + proposal_areas  # (n_old, n_new)

        moved_old_box_proposals = np.argmax(scores, axis=1)

        score_temp = scores.copy()
        new_proposed_score = np.zeros(max((len(self._current_boxes), len(proposal_boxes)))) - float('inf')

        bachelors = np.ones(len(proposal_boxes), dtype=bool)
        for k in range(min(len(self._current_boxes), len(proposal_boxes))):
            i, j = np.unravel_index(np.argmax(np.ravel(score_temp)), score_temp.shape)
            new_proposed_score[i] = score_temp[i, j]
            score_temp[i, :] = -float('inf')
            score_temp[:, j] = -float('inf')
            bachelors[j]=False

        if len(self._current_boxes) > len(proposal_boxes):
            updated_scores = np.maximum(self._old_score, new_proposed_score)
        else:
            # new_scores = np.zeros(len(proposal_boxes))
            # new_scores[bachelors] = proposal_areas[bachelors]
            # new_scores = np.concatenate([new_proposed_score, proposal_areas[bachelors]])
            update_scores = np.maximum(np.concatenate([self._old_score, proposal_areas[bachelors]]), new_proposed_score)

        self._old_score = update_scores
        # if self._old_score larger threashold:
        #     return those boundingboxes


        # Do comparison here
        #
        #     score_temp[np.isnan(new_proposed_score)] = np.argmax(score_temp, axis=1)
        # else:
        #     score_temp[np.isnan(new_proposed_score)] = np.argmax(score_temp, axis=0)





        proposal_boxes = np.array(proposal_boxes)
        new_bb_areas = proposal_boxes[:, 2] * proposal_boxes[:, 3]
        overlaps = bb_overlap(self._current_boxes, proposal_boxes)
        new_scores = self._old_score * overlaps + new_bb_areas
        if np.max(new_scores) > self._old_score:
            selected_bbox = np.argmax(new_scores)
            out_box = proposal_boxes[selected_bbox]
            self._old_score = new_scores[selected_bbox]
            self._current_boxes = out_box
        else:
            out_box = self._current_boxes

        return [out_box]


class BoundingBoxPIDSmoother(object):

    def __init__(self, k_p=1, k_i=0.1, k_d=.1):
        self.position = None
        self.k_i = np.array(k_i, dtype=float)
        self.k_p = np.array(k_p, dtype=float)
        self.k_d = np.array(k_d, dtype=float)
        self.integral = 0
        self.last_error = 0

    def __call__(self, bounding_boxes):
        if len(bounding_boxes)==0:
            return bounding_boxes
        assert len(bounding_boxes)==1
        bbox = bounding_boxes[0]
        if self.position is None:
            self.position = bbox
        error = bbox - self.position
        self.integral = self.integral + error
        deriv = error - self.last_error
        self.last_error = error
        control_signal = self.k_p*error + self.k_i*self.integral + self.k_d*deriv
        self.position = self.position + control_signal
        return [self.position.copy()]


def bb_overlap(bounding_box, candidates):
    """
    :param bounding_box: A shape: (4, ) array.. (left, top, width, height)
    :param candidates: A shape (n_samples, 4) array
    :return: A list of numbers in [0, 1] indicating the overlap ratio (intersection/union)
    """
    bounding_box = np.asarray(bounding_box)
    candidates = np.asarray(candidates)

    ixmin = np.maximum(candidates[..., 0], bounding_box[0])
    iymin = np.maximum(candidates[..., 1], bounding_box[1])
    ixmax = np.minimum(candidates[..., 2] + candidates[..., 0]-1, bounding_box[2]+bounding_box[0]-1)
    iymax = np.minimum(candidates[..., 3] + candidates[..., 1]-1, bounding_box[3]+bounding_box[1]-1)
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    intersection = iw * ih

    overlap_ratio = intersection / (bounding_box[2]*bounding_box[3] + candidates[..., 2]*candidates[..., 3] - intersection)

    return overlap_ratio


class FaceDetectorExtractor(object):

    def __init__(self, detector, bb_pipeline = (), crop_extension=0):
        self._memo_hashable = locals()
        del self._memo_hashable['self']
        self.crop_extension = crop_extension
        self.detector = detector
        self.bb_pipeline = bb_pipeline

    def __call__(self, image):
        face_bounding_boxes = self.detector(image)
        for f in self.bb_pipeline:
            face_bounding_boxes = f(face_bounding_boxes)
        face_crops = [crop_face(image, f, extend=self.crop_extension) for f in face_bounding_boxes]
        return face_crops

    def memo_hashable(self):
        return self._memo_hashable


def crop_face(im, bbox, extend=0):
    left, top, width, height = bbox
    (left, top, width, height) = get_extended_crop((left, top, width, height), extend=extend)
    sub_im = im[max(0, top): min(top+height, im.shape[0]), max(0, left):min(left+width, im.shape[1])]
    return sub_im


def get_extended_crop(bbox, extend=0):
    left, top, width, height = bbox
    new_width = int(np.round((1+extend) * width))
    new_height = int(np.round((1+extend) * height))
    new_left = int(np.round(left - 0.5*extend*width))
    new_top = int(np.round(top - 0.5*extend*height))
    return new_left, new_top, new_width, new_height


def get_haarcascade_folder():
    root = input("Enter the directory where the haar cascade xml files are stored.  This will be saved and can be changed in the ~/.quvacoderc file. >> ")
    assert os.path.isdir(root), '"%s" is not a directory.' % (root, )
    return root


def get_face_detector_version(version = 'default'):
    """
    Here we can save versions of the face detector.  Hopefully this prevents the proliferation of different
    experimental versions scattered all over the code.  If you have a configuration that you want to keep,
    put it here.
    :param version: A string identifying the version of the face detector
    :return: A FaceDetectorExtractor object which can be called as:
        faces = face_detector(image)
        Where:
            image is a (size_y, size_x, 3) RGB or RGB image
            faces is a list of (size_y, size_x, 3) images of "faces" found.
    """
    face_detector = {
        'default': lambda: FaceDetectorExtractor(
            detector=FaceBoundingBoxExtractor(template = 'haarcascade_frontalface_alt', min_relative_size = 0.1, face_detector_args=dict(scaleFactor = 1.05, minNeighbors=3)),
            crop_extension=0.1
            ),
        # 'video': lambda: FaceDetectorExtractor(template = 'haarcascade_frontalface_alt', crop_extension=0.1, min_relative_size = 0.1, face_detector_args=dict(scaleFactor = 1.05, minNeighbors=1))
        'video': lambda: FaceDetectorExtractor(
            detector=FaceBoundingBoxExtractor(template = 'haarcascade_frontalface_alt', min_relative_size = 0.1, face_detector_args=dict(scaleFactor = 1.2, minNeighbors=3)),
            bb_pipeline = [SingleFaceBoundingBox(memory = 0.7), BoundingBoxPIDSmoother(k_p=.7, k_i=0.005, k_d=-.2)],
            # bb_pipeline = [SingleFaceBoundingBox(memory = 0.7), BoundingBoxPIDSmoother(k_p=.0, k_i=.81, k_d=.81)],
            # bb_pipeline = [SingleFaceBoundingBox(memory = 0.7), BoundingBoxPIDSmoother(k_p=.8)],
            crop_extension=0.1
            ),
        # 'mijung': lambda: MijungFaceDetector()
        }[version]()
    return face_detector
