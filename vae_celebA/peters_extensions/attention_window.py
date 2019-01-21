import cv2
import numpy as np

from artemis.general.mymath import argmaxnd
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from vae_celebA.image_utils.video_camera import VideoCamera


class BackgroundMaskMOG(object):

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def __call__(self, im):
        fgmask = self.fgbg.apply(im)
        return fgmask


def select_most_interesting_region(attention_image, region_size):
    """
    :param image: A (height, width) image
    :param region_size: (width, height) of region
    :return: A (left, top, right, bottom) bounding box.
    """

    srx, sry = region_size
    cumsum_activity = np.abs(attention_image).cumsum(axis=1).cumsum(axis=0)
    activity_in_box = cumsum_activity[sry:, srx:] - cumsum_activity[sry:, :-srx] - cumsum_activity[:-sry, srx:] + cumsum_activity[:-sry, :-srx]
    max_row, max_col = argmaxnd(activity_in_box)
    return max_col, max_row, max_col+srx, max_row+sry


class AttentionWindow:

    def __init__(self, window_size = (256, 256), decay_rate=0.3, downsample_to = (80, 60), boredom_decay = 0., region_pad_factor = (.5, .5)):
        """

        :param window_size:
        :param decay_rate:
        :param downsample_to:
        :param boredom_decay: (Doesnt really work now, just causes rapid oscillation when 2 things are present... needs some kind of rise-time).
        :param background_masker:
        :param region_pad_factor:
        """
        assert 0<decay_rate<=1
        self.decay_rate = decay_rate
        self.downsample_to = downsample_to
        self.masker = None  # Just for the sake of pickling.
        self.window_size = window_size
        self.attention_map = None
        self.region_pad_factor = region_pad_factor
        self.boredom_decay = boredom_decay

    @classmethod
    def from_default_settings(cls, default_name = 'lab'):
        if default_name=='lab':
            return AttentionWindow(window_size=(200, 200), decay_rate=0.5, downsample_to=(80, 60), boredom_decay=0.0)
        else:
            raise NotImplementedError(default_name)

    def get_attention_window_and_crop(self, image):

        if self.masker is None:
            self.masker = BackgroundMaskMOG()

        window_size = self.window_size if isinstance(self.window_size[0], int) else (int(image.shape[1]*self.window_size[1]+.5), int(image.shape[0]*self.window_size[0]+.5))

        ds_img = cv2.resize(image, self.downsample_to) if self.downsample_to is not None else image

        fgmask = self.masker(ds_img)
        if self.attention_map is None:
            self.attention_map = np.zeros(fgmask.shape)
        self.attention_map = self.attention_map*(1-self.decay_rate) + fgmask*(self.decay_rate/256)

        x_factor, y_factor = (image.shape[1]/self.downsample_to[0], image.shape[0]/self.downsample_to[1]) if self.downsample_to is not None else (1., 1.)
        region_size=(int(window_size[0]/(x_factor*(1+self.region_pad_factor[0]))+.5), int(window_size[1]/(y_factor*(1+self.region_pad_factor[1]))+.5))
        l, t, r, b = select_most_interesting_region(self.attention_map, region_size=region_size)

        if self.boredom_decay:
            self.attention_map[t:b, l:r] *= (1-self.boredom_decay)

        l2, t2, r2, b2 = _center_to_bbox(old_im_center=[(l+r)/2., (t+b)/2.], old_im_size=[ds_img.shape[1], ds_img.shape[0]], new_im_size=[image.shape[1], image.shape[0]], new_bbox_size=window_size)
        crop = image[t2:b2, l2:r2]

        return self.attention_map, crop, (l2, t2, r2, b2)

    def __call__(self, image):
        _, crop, _ = self.get_attention_window_and_crop(image)
        return crop


def _center_to_bbox(old_im_center, old_im_size, new_im_size, new_bbox_size):

    l, t = [max(0, min(new_dim-new_box_size, int(new_dim * old_center/old_dim+.5) - new_box_size//2)) for old_dim, old_center, new_dim, new_box_size in zip(old_im_size, old_im_center, new_im_size, new_bbox_size)]
    b, r = [start+new_box_size for start, new_box_size in zip([t, l], new_bbox_size)]
    return l, t, r, b


if __name__ == '__main__':

    FAST = True
    windower = AttentionWindow(window_size=(200, 200), decay_rate=0.5, downsample_to=(80, 60), boredom_decay=0.0)

    if FAST:
        for im in VideoCamera(size=(640, 480), device=0, mode='bgr').iterator():
            im = im[:192]
            crop = windower(im)
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

