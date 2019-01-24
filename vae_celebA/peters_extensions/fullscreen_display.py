import cv2
import numpy as np


_FULL_FRAMES = {}


def show_fullscreen_v1(image, background_colour = None, window_name='window', display_number = 0, display_sizes=None):
    """
    Draw a fullscreen image.

    :param image: The image to show.
        If integer, it will be assumed to be in range [0..255]
        If float, it will be assumed to be in range [0, 1]
    :param background_colour: The background colour, as a BGR tuple.
    :param window_name: Name of the window (can be used to draw multiple fullscreen windows)
    :param display_number: Which monitor to display to.
    :param display_sizes: Size of displays (needed only if adding a background colour)
    """
    if image.dtype=='float':
        image = (image*255.999).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    if image.ndim==2:
        image = image[:, :, None]

    assert display_number in (0, 1), 'Only 2 displays supported for now.'
    if window_name not in _FULL_FRAMES:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if display_number == 1:
            assert display_sizes is not None
            first_display_size = display_sizes[0]
            cv2.moveWindow(window_name, *first_display_size)
        cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
        if background_colour is not None:
            background_colour = np.array(background_colour)
            if background_colour.dtype=='int':
                background_colour = background_colour.astype(np.uint8)
            else:
                background_colour = (background_colour*255.999).astype(np.uint8)
            assert display_sizes is not None, "Unfortunately, if you want to specify background color you need to specify display sizes."
            pic_display_size = display_sizes[display_number]
            aspect_ratio = pic_display_size[1]/float(pic_display_size[0])  # (hori/vert)
            frame_size_x = int(max(image.shape[0]/aspect_ratio, image.shape[1]))
            frame_size_y = int(max(image.shape[1]*aspect_ratio, image.shape[0]))
            _FULL_FRAMES[window_name] = np.zeros((frame_size_y, frame_size_x, 3), dtype=np.uint8) + background_colour
        else:
            _FULL_FRAMES[window_name] = None

    if _FULL_FRAMES[window_name] is not None:
        frame = _FULL_FRAMES[window_name]
        start_y, start_x = (frame.shape[0] - image.shape[0])//2, (frame.shape[1] - image.shape[1])//2
        frame[start_y: start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
        display_img = frame
    else:
        display_img = image

    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)


def show_fullscreen(image, background_colour = None, window_name='window', display_number = 0, display_sizes=None):
    """
    Draw a fullscreen image.

    :param image: The image to show.
        If integer, it will be assumed to be in range [0..255]
        If float, it will be assumed to be in range [0, 1]
    :param background_colour: The background colour, as a BGR tuple.
    :param window_name: Name of the window (can be used to draw multiple fullscreen windows)
    :param display_number: Which monitor to display to.
    :param display_sizes: Size of displays (needed only if adding a background colour)
    """
    if image.dtype=='float':
        image = (image*255.999).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    if image.ndim==2:
        image = image[:, :, None]

    assert display_number in (0, 1), 'Only 2 displays supported for now.'
    if window_name not in _FULL_FRAMES:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if display_number == 1:
            assert display_sizes is not None
            first_display_size = display_sizes[0]
            cv2.moveWindow(window_name, *first_display_size)
        cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
        if background_colour is not None:
            background_colour = np.array(background_colour)
            if background_colour.dtype=='int':
                background_colour = background_colour.astype(np.uint8)
            else:
                background_colour = (background_colour*255.999).astype(np.uint8)
            assert display_sizes is not None, "Unfortunately, if you want to specify background color you need to specify display sizes."
            pic_display_size = display_sizes[display_number]
            aspect_ratio = pic_display_size[1]/float(pic_display_size[0])  # (hori/vert)
            frame_size_x = int(max(image.shape[0]/aspect_ratio, image.shape[1]))
            frame_size_y = int(max(image.shape[1]*aspect_ratio, image.shape[0]))
            _FULL_FRAMES[window_name] = np.zeros((frame_size_y, frame_size_x, 3), dtype=np.uint8) + background_colour
        else:
            _FULL_FRAMES[window_name] = None

    if _FULL_FRAMES[window_name] is not None:
        frame = _FULL_FRAMES[window_name]
        start_y, start_x = (frame.shape[0] - image.shape[0])//2, (frame.shape[1] - image.shape[1])//2
        frame[start_y: start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
        display_img = frame
    else:
        display_img = image

    display_img = cv2.resize(display_img, display_sizes[display_number])

    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)


if __name__ == '__main__':
    for t in np.linspace(0, 10, 1000):
        im = np.sin(-4*t+np.sin(t/4.)*sum(xi**2 for xi in np.meshgrid(*[np.linspace(-20, 20, 480)]*2)))*.5+.5
        show_fullscreen(im, background_colour=(0, 0, 0), display_sizes=[(1440, 900), (1920, 1080)], display_number=0)  #
        # show_fullscreen(im, background_colour=None, display_number=0)
