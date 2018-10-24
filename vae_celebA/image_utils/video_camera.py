import cv2
import time

__author__ = 'peter'

#
class VideoCamera(object):

    def __init__(self, size = None, device=0):
        """
        :param size: Optionally, a 2-tuple of (width, height)
        :return:
        """
        self.camera = cv2.VideoCapture(device)
        if size is not None:
            width, height = size
            if cv2.__version__.startswith('2'):
                self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

            elif cv2.__version__.startswith('3'):
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._missed_frame_count = 0

    def read(self):
        """
        Read the image
        :return: Either a (size_y, size_x, 3) ndarray containing the BGR image, or None if the image cannot be read.
        """
        retval, bgr_im = self.camera.read()
        return bgr_im

    def iterator(self, missed_frame_sleep_time = 0.1):
        while True:
            im = self.read()
            if im is None:
                self._missed_frame_count += 1
                print("Missed Camera Frame for the %s'th time!" % (self._missed_frame_count, ))
                time.sleep(missed_frame_sleep_time)
            else:
                yield im



# class VideoCamera(object):
#
#     def __init__(self, size = None):
#         """
#         :param size: Optionally, a 2-tuple of (width, height)
#         :return:
#         """
#         self.camera = cv2.VideoCapture(0)
#         if size is not None:
#             width, height = size
#             if cv2.__version__.startswith('2'):
#                 self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
#                 self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
#
#             elif cv2.__version__.startswith('3'):
#                 self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#                 self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self._missed_frame_count = 0
#
#     def read(self):
#         """
#         Read the image
#         :return: Either a (size_y, size_x, 3) ndarray containing the BGR image, or None if the image cannot be read.
#         """
#         retval, bgr_im = self.camera.read()
#         return bgr_im
#
#     def iterator(self, missed_frame_sleep_time = 0.1):
#         last_im = None
#         while True:
#             im = self.read()
#             if im is None:
#                 if last_im is None:
#                     self._missed_frame_count += 1
#                     print "Missed Camera Frame for the %s'th time!" % (self._missed_frame_count, )
#                     time.sleep(missed_frame_sleep_time)
#                 else:
#                     yield last_im
#             else:
#                 last_im = im

