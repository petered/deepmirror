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


def select_most_interesting_region(attention_image, region_size, rel_coords = False):
    """
    :param image: A (height, width) image
    :param region_size: (width, height) of region
    :return: A (left, top, right, bottom) bounding box.
    """

    srx, sry = region_size if not rel_coords else (round(attention_image.shape[1]*region_size[0]), round(attention_image.shape[0]*region_size[1]))
    cumsum_activity = np.abs(attention_image).cumsum(axis=1).cumsum(axis=0)
    activity_in_box = cumsum_activity[sry:, srx:] - cumsum_activity[sry:, :-srx] - cumsum_activity[:-sry, srx:] + cumsum_activity[:-sry, :-srx]
    max_row, max_col = argmaxnd(activity_in_box)
    if rel_coords:
        return float(max_col/attention_image.shape[1]), float(max_row/attention_image.shape[0]), float((max_col+srx)/attention_image.shape[1]), float((max_row+sry)/attention_image.shape[0])
    else:
        return max_col, max_row, max_col+srx, max_row+sry


class AttentionWindow:

    def __init__(self, window_size = (256, 256), decay_rate=0.3, downsample_to = (80, 60), boredom_decay = 0., activity_box = (.25, .25, .75, .75)):
        """

        :param window_size: The size of the attention window to return
        :param decay_rate:
        :param downsample_to:
        :param boredom_decay: (Doesnt really work now, just causes rapid oscillation when 2 things are present... needs some kind of rise-time).
        :param background_masker:
        :param region_pad_factor: An (L, T, R, B) tuple that identifies the box within the window in which to search
            for activity.  For example the default of (0.25, 0.25, 0.75, 0.75) means return the window with the maximum
            activity in the middle.  The dimensions box can also be greater than one if you want to specify for instance
            that you want the "activity search" box to lie outside the attention box.  For detecting faces, a good box
            may be (0.25, 0.25, 0.75, 1.5) ... Because the face is typically on top of a body which is moving around.
        """
        assert 0<decay_rate<=1
        self.decay_rate = decay_rate
        self.downsample_to = downsample_to
        self.masker = None  # Just for the sake of pickling.
        self.window_size = window_size
        self.attention_map = None
        self.activity_box = activity_box
        self.boredom_decay = boredom_decay

    @classmethod
    def from_default_settings(cls, default_name = 'lab'):
        if default_name=='lab':
            return AttentionWindow(window_size=(200, 200), decay_rate=1, downsample_to=(80, 60), boredom_decay=0.0)
        elif default_name=='faces':
            return AttentionWindow(window_size=(128, 256), activity_box=(0.1, 0.5, 0.9, 1.25), decay_rate=1, downsample_to=(80, 60), boredom_decay=0.0)
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

        # Scale factor relating downsampled to image dims
        # x_factor, y_factor = (image.shape[1]/self.downsample_to[0], image.shape[0]/self.downsample_to[1]) if self.downsample_to is not None else (1., 1.)

        # Relative height and width of window
        w_win, h_win = window_size[0]/image.shape[1], window_size[1]/image.shape[0]

        # Get the (relative) size of the attention window to search:
        l_box, t_box, r_box, b_box = self.activity_box
        w_box, h_box = (r_box-l_box), (b_box-t_box)
        region_size = w_win*w_box, h_win*h_box

        # Find the box
        l_att, t_att, r_att, b_att = select_most_interesting_region(self.attention_map, region_size=region_size, rel_coords=True)

        # Adjust the relative "attention box" coordinates to the relative "window" coordinates
        # w_att, h_att = (r_att-l_att), (b_att - t_att)
        rel_bbox = l_att-l_box*w_win, t_att-t_box*h_win, r_att+(1-r_box)*w_win, b_att+(1-b_box)*h_win

        l, t, r, b = rel_to_real_bbox(im_size = (image.shape[1], image.shape[0]), rel_bbox=rel_bbox)

        if self.boredom_decay:
            self.attention_map[t:b, l:r] *= (1-self.boredom_decay)

        # l2, t2, r2, b2 = _center_to_bbox(old_im_center=[(l+r)/2., (t+b)/2.], old_im_size=[ds_img.shape[1], ds_img.shape[0]], new_im_size=[image.shape[1], image.shape[0]], new_bbox_size=window_size)
        crop = image[t:b, l:r]

        return self.attention_map, crop, (l, t, r, b)

    def __call__(self, image):
        _, crop, _ = self.get_attention_window_and_crop(image)
        return crop


def rel_to_real_bbox(im_size, rel_bbox):
    l_rel, t_rel, r_rel, b_rel = rel_bbox
    w_rel, h_rel = (r_rel-l_rel), (b_rel-t_rel)
    w_im, h_im = im_size
    w_box, h_box = round(w_rel*w_im), round(h_rel*h_im)
    l = max(0, min(w_im-w_box, round(l_rel*w_im)))
    t = max(0, min(h_im-h_box, round(t_rel*h_im)))
    return l, t, l+w_box, t+h_box


def crop_image_with_rel_box(im, rel_bbox):
    l, t, r, b = rel_to_real_bbox(im_size = (im.shape[1], im.shape[0]), rel_bbox=rel_bbox)
    return im[t:b, l:r]


def _center_to_bbox(old_im_center, old_im_size, new_im_size, new_bbox_size):

    l, t = [max(0, min(new_dim-new_box_size, int(new_dim * old_center/old_dim+.5) - new_box_size//2)) for old_dim, old_center, new_dim, new_box_size in zip(old_im_size, old_im_center, new_im_size, new_bbox_size)]
    b, r = [start+new_box_size for start, new_box_size in zip([t, l], new_bbox_size)]
    return l, t, r, b


if __name__ == '__main__':

    FAST = False
    # windower = AttentionWindow(window_size=(200, 200), decay_rate=0.5, downsample_to=(80, 60), boredom_decay=0.0)
    windower = AttentionWindow.from_default_settings('faces')

    if FAST:
        for im in VideoCamera(size=(640, 480), device=0, mode='bgr').iterator():
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
