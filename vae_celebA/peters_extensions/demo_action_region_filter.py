import time
import numpy as np


def argpercentile(vec, percentiles):
    """
    Given a positive vector x, return indices at which the given percentiles are crossed.
    """
    assert np.all(vec>=0) and vec.ndim==1
    cs = np.cumsum(vec)

    # dbplot(cs, plot_type='line')

    total = cs[-1]
    # vals = np.percentile(cs, percentiles)
    # print(vals)

    ixs = np.argmax(cs>total*percentiles[0]/100), np.argmax(cs>total*percentiles[1]/100)

    # ixs = np.searchsorted(cs, vals)
    return ixs



class ActionRegionFilter(object):
    """
    Filter out the most action-packed part of the image
    """

    def __init__(self, time_constant = 20, units='frames', subsampling=None, action_fraction = 0.6):
        assert units in ('frames', 'sec')
        assert 0 <= action_fraction <= 1
        self.avg = None
        self.units = units
        self.time_constant = time_constant
        self.subsampling = subsampling
        self.percentiles = (.5-action_fraction/2.)*100, (.5+action_fraction/2.)*100
        self.last_time = -float('inf')

    def __call__(self, img):

        working_im = img[::self.subsampling, ::self.subsampling] if self.subsampling is not None else img

        if self.avg is None:
            self.avg = working_im
            return img
        else:
            if self.units == 'sec':
                current_time = time.time()
                frac = np.exp(-(current_time - self.last_time)/self.time_constant)
                self.last_time = current_time
            else:
                frac = 1./self.time_constant
            self.avg = (1-frac)*self.avg + frac*working_im

            changes = np.sum(np.abs(working_im - self.avg), axis=2)

            dbplot(changes, 'changes')
            dbplot(changes.sum(axis=0), 'hchanges', plot_type='line')

            hs, he = argpercentile(changes.sum(axis=0), percentiles=self.percentiles)
            vs, ve = argpercentile(changes.sum(axis=1), percentiles=self.percentiles)

            if self.subsampling is not None:
                hs, he = hs*self.subsampling, he*self.subsampling
                vs, ve = hs*self.subsampling, he*self.subsampling

            print((hs, he))
            return img[vs:ve+1, hs:he+1]


if __name__ == '__main__':
    import itertools
    from vae_celebA.image_utils.video_camera import VideoCamera
    from artemis.plotting.db_plotting import dbplot, hold_dbplots
    import cv2
    arf = ActionRegionFilter(time_constant = 20, action_fraction=0.9, subsampling=2)

    cam = VideoCamera(size=(640, 480), device=0)

    for t in itertools.count(0):
        im = cam.read()
        if im is not None:
            fim = arf(im)
            with hold_dbplots():
                dbplot(im, 'im')
                dbplot(fim, 'fim')