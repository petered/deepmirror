import cv2
import os
import numpy as np
from artemis.fileman.config_files import get_config_value
import pickle
from artemis.fileman.images2gif import readGif
from artemis.fileman.local_dir import get_local_path
import re


# class FaceExtractor(object):
#     def __init__(self, template = 'haarcascade_frontalface_default', eye_filter = False, crop_extension = 0, face_detector_args = {}):
#         """
#         :param template: One of: {'haarcascade_frontalface_default', '9,563 haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt_tree'}
#             Listed in decreasing order of recall rate (increasing order of precision)
#         :param eye_filter:
#         :param crop_extension:
#         :param face_detector_args:
#         :return:
#         """
#
#         # cvroot = get_config_value('.quvacoderc', section='cv', option='root', default_generator=get_cv_root, write_default=True)
#
#         haarcascade_folder = get_config_value('.quvacoderc', section='cv', option='haarcascades', default_generator=get_haarcascade_folder, write_default=True)
#
#         # TODO: Better way of locating
#         self.face_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_folder, "%s.xml" % (template, )))
#         if eye_filter:
#             self.eye_cascade = cv2.CascadeClassifier(os.path.join(haarcascade_folder, "haarcascade_eye.xml"))
#         self.eye_augmented = eye_filter
#         if self.eye_augmented:
#             from artemis.plotting.db_plotting import dbplot
#         self.face_detector_args = face_detector_args
#         self.crop_extension = crop_extension
#
#     def __call__(self, im):
#         """
#         :param im: A (size_y, size_x, 3) RGB image (or BRG if already_bgr is True)
#         :param already_bgr: Boolean, indicating if the image is already in BGR mode.
#         :return: A list of (crop_size_y, crop_size_x, 3) arrays containing cropped faces
#         """
#         gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         face_bounding_boxes = self.face_cascade.detectMultiScale(gray_im, **self.face_detector_args)
#         face_crops = [crop_face(im, f, extend=self.crop_extension) for f in face_bounding_boxes]
#
#         if self.eye_augmented:
#             face_crops = [f for f, bb in zip(face_crops, face_bounding_boxes) if self._test_for_eyes(crop_face(gray_im, bb))]
#         return face_crops
#
#     def _test_for_eyes(self, grey_face):
#         eyes = self.eye_cascade.detectMultiScale(grey_face)
#         return len(eyes)==2


# def get_extended_crop((left, top, width, height), extend=0):
#     new_width = np.round((1+extend) * width)
#     new_height = np.round((1+extend) * height)
#     new_left = np.round(left - 0.5*extend*width)
#     new_top = np.round(top - 0.5*extend*height)
#     return new_left, new_top, new_width, new_height
#
#
# def crop_face(im, (left, top, width, height), extend=0):
#     (left, top, width, height) = get_extended_crop((left, top, width, height), extend=extend)
#     sub_im = im[top: top+height, left:left+width]
#     return sub_im


# def get_haarcascade_folder():
#     root = raw_input("Enter the directory where the haar cascade xml files are stored.  This will be saved and can be changed in the ~/.quvacoderc file. >> ")
#     assert os.path.isdir(root), '"%s" is not a directory.' % (root, )
#     return root



#
# def is_url(path):
#     regex = re.compile(
#         r'^(?:http|ftp)s?://' # http:// or https://
#         r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
#         r'localhost|' #localhost...
#         r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
#         r'(?::\d+)?' # optional port
#         r'(?:/?|[/?]\S+)$', re.IGNORECASE)
#     return True if re.match(regex, path) else False
#
# def smart_load(relative_path, use_cache = False):
#     """
#     Load a file, with the method based on the extension.  See smart_save doc for the list of extensions.
#     :param relative_path: Path, relative to your data directory.  The extension determines the type of object you will load.
#     :return: An object, whose type depends on the extension.  Generally a numpy array for data or an object for pickles.
#     """
#
#     # TODO... Support for local files, urls, etc...
#     its_a_url = is_url(relative_path)
#     _, ext = os.path.splitext(relative_path)
#     if its_a_url:
#         if use_cache:
#             local_path = get_file_and_cache(relative_path)
#         else:
#             local_path = get_temp_file(relative_path)
#     else:
#         local_path = get_local_path(relative_path)
#     if ext=='.pkl':
#         with open(local_path) as f:
#             obj = pickle.load(f)
#     elif ext=='.gif':
#         obj = np.array(readGif(local_path))
#     elif ext in ('.jpg', '.jpeg', '.png', '.JPG'):
#         from PIL import Image
#         pic = Image.open(local_path)
#         pic_arr = np.array(pic.getdata(), dtype='uint8')
#         if pic_arr.size == np.prod(pic.size):  # BW image
#             pix = pic_arr.reshape(pic.size[1], pic.size[0])
#         elif pic_arr.size == np.prod(pic.size)*3:  # RGB image
#             pix = pic_arr.reshape(pic.size[1], pic.size[0], 3)
#         elif pic_arr.size == np.prod(pic.size)*4:  # RGBA image... just take RGB for now!
#             pix = pic_arr.reshape(pic.size[1], pic.size[0], 4)[:, :, :3]
#         else:
#             raise Exception("Pixel count: %s, did not divide evenly into picture size: %s" % (pic_arr.size, pic.size))
#         return pix
#     else:
#         raise Exception("No method exists yet to load '%s' files.  Add it!" % (ext, ))
#     if its_a_url:
#         os.remove(local_path)  # Clean up after
#     return obj
