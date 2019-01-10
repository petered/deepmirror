import numpy as np
import cv2

from artemis.general.checkpoint_counter import do_every
from artemis.general.ezprofile import profile_context, get_profile_contexts_string
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from vae_celebA.image_utils.video_camera import VideoCamera


class OpticFlowCalculator:

    def __init__(self, **kwargs):

        self.last = None

    def __call__(self, im):

        if im.ndim==3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        if self.last is None:
            self.last = im

        flow = cv2.calcOpticalFlowFarneback(self.last, im, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.last = im
        return flow



class BackgroundSubtractor:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def __call__(self, img):
        fgmask = self.fgbg.apply(img)
        return fgmask






if __name__ == '__main__':

    cam = VideoCamera(mode='rgb', size=(160, 120))
    bgsub = BackgroundSubtractor()

    for im in cam.iterator():

        # for i in range(im.shape[-1]):
        #     im[:, :, i] = cv2.equalizeHist(im[:, :, i])
        with profile_context('flow'):
            bgless_img = bgsub(im)
        with hold_dbplots():
            dbplot(im, 'image')
            # dbplot(np.sum(np.abs(flow), axis=-1), 'flow')
            dbplot(bgless_img, 'flow')
        if do_every('5s'):
            print(get_profile_contexts_string())
