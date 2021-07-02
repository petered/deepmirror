from collections import OrderedDict
from typing import Union, Hashable, Dict, Tuple, List, Set, Optional
import numpy as np
from attr import attrib, attrs
import cv2

DEFAULT_GAP_COLOR = np.array((20, 20, 20), dtype=np.uint8)
BGRImageArray = 'Array[H,W,3:uint8]'
DEFAULT_WINDOW_NAME = 'Window'


class BGRColors:
    """ An Enum of Colors as (Blue, Green, Red) tuples"""

    BLUE = 255, 0, 0
    GREEN = 0, 255, 0
    WHITE = 255, 255, 255
    BLACK = 0, 0, 0
    DARK_GRAY = 50, 50, 50
    VERY_DARK_GRAY = 20, 20, 20


class InputTimeoutError(Exception):
    """ Raised if you have not received user input within the timeout """
    pass


class Keys(object):
    """
    An enum identifying keys on the keyboard
    """
    NONE = None  # Code for "no key press"
    RETURN = 'RETURN'
    SPACE = 'SPACE'
    DELETE = 'DELETE'
    LSHIFT = 'LSHIFT'
    RSHIFT = 'RSHIFT'
    TAB = 'TAB'
    ESC = "ESC"
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    UP = 'UP'
    DOWN = 'DOWN'
    DASH = 'DASH'
    EQUALS = 'EQUALS'
    BACKSPACE = 'BACKSPACE'
    LBRACE = 'LBRACE'
    RBRACE = 'RBRACE'
    BACKSLASH = 'BACKSLASH'
    SEMICOLON = 'SEMICOLON'
    APOSTROPHE = 'APOSTROPHE'
    COMMA = 'COMMA'
    PERIOD = 'PERIOD'
    SLASH = 'SLASH'
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'
    H = 'H'
    I = 'I'
    J = 'J'
    K = 'K'
    L = 'L'
    M = 'M'
    N = 'N'
    O = 'O'
    P = 'P'
    Q = 'Q'
    R = 'R'
    S = 'S'
    T = 'T'
    U = 'U'
    V = 'V'
    W = 'W'
    X = 'X'
    Y = 'Y'
    Z = 'Z'
    n0 = '0'
    n1 = '1'
    n2 = '2'
    n3 = '3'
    n4 = '4'
    n5 = '5'
    n6 = '6'
    n7 = '7'
    n8 = '8'
    n9 = '9'
    np0 = '0'
    np1 = '1'
    np2 = '2'
    np3 = '3'
    np4 = '4'
    np5 = '5'
    np6 = '6'
    np7 = '7'
    np8 = '8'
    np9 = '9'
    UNKNOWN = 'UNKNOWN'


_keydict = {
    # On a MAC these are the key codes
    -1: Keys.NONE,
    27: Keys.ESC,
    13: Keys.RETURN,
    32: Keys.SPACE,
    255: Keys.DELETE,
    225: Keys.LSHIFT,
    226: Keys.RSHIFT,
    9: Keys.TAB,
    81: Keys.LEFT,
    82: Keys.UP,
    83: Keys.RIGHT,
    84: Keys.DOWN,
    45: Keys.DASH,
    61: Keys.EQUALS,
    8: Keys.BACKSPACE,
    91: Keys.LBRACE,
    93: Keys.RBRACE,
    92: Keys.BACKSLASH,
    59: Keys.SEMICOLON,
    39: Keys.APOSTROPHE,
    44: Keys.COMMA,
    46: Keys.PERIOD,
    47: Keys.SLASH,
    63: Keys.SLASH,  # On on thinkpad at least
    97: Keys.A,
    98: Keys.B,
    99: Keys.C,
    100: Keys.D,
    101: Keys.E,
    102: Keys.F,
    103: Keys.G,
    104: Keys.H,
    105: Keys.I,
    106: Keys.J,
    107: Keys.K,
    108: Keys.L,
    109: Keys.M,
    110: Keys.N,
    111: Keys.O,
    112: Keys.P,
    113: Keys.Q,
    114: Keys.R,
    115: Keys.S,
    116: Keys.T,
    117: Keys.U,
    118: Keys.V,
    119: Keys.W,
    120: Keys.X,
    121: Keys.Y,
    122: Keys.Z,
    48: Keys.n0,
    49: Keys.n1,
    50: Keys.n2,
    51: Keys.n3,
    52: Keys.n4,
    53: Keys.n5,
    54: Keys.n6,
    55: Keys.n7,
    56: Keys.n8,
    57: Keys.n9,
    158: Keys.n0,
    156: Keys.np1,
    153: Keys.np2,
    155: Keys.np3,
    150: Keys.np4,
    157: Keys.np5,
    152: Keys.np6,
    149: Keys.np7,
    151: Keys.np8,
    154: Keys.np9,
}


@attrs
class TextDisplayer:
    """ Converts text to image """
    text_color = attrib(default=BGRColors.WHITE)
    thickness = attrib(default=1)
    font = attrib(default=cv2.FONT_HERSHEY_PLAIN)
    scale = attrib(default=1)
    background_color = attrib(factory=lambda: DEFAULT_GAP_COLOR)
    size = attrib(type=Optional[Tuple[int, int]], default=None)  # (width, height) in characters
    vspace = attrib(type=float, default=0.4)
    expand_box = attrib(type=bool, default=True)
    _last_size = attrib(type=Optional[Tuple[int, int]], default=None)

    def render(self, data: str) -> BGRImageArray:

        lines = data.split('\n')
        longest_line = max(lines, key=len)
        (text_width, text_height), baseline = cv2.getTextSize(longest_line, self.font, self.scale, self.thickness)

        width, height = text_width + 10, int(len(lines) * text_height * (1 + self.vspace))

        if self.expand_box:
            oldwidth, oldheight = self._last_size if self._last_size is not None else (0, 0)
            self._last_size = max(oldwidth, width), max(oldheight, height)
            width, height = self._last_size
        img = create_gap_image((width, height) if self.size is None else self.size, gap_colour=self.background_color)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (0, int(baseline * 2 + i * (1 + self.vspace) * text_height)), fontFace=self.font, fontScale=self.scale, color=self.text_color,
                        thickness=self.thickness, bottomLeftOrigin=False)
        return img


def cvkey_to_key(cvkeycode):
    """
    Given a cv2 keycode, return the key, which will be a member of the Keys enum.
    :param cvkeycode: The code returned by cv2.waitKey
    :return: A string, one of the members of Keys
    """
    key = _keydict.get(cvkeycode & 0xFF if cvkeycode>0 else cvkeycode, Keys.UNKNOWN)  # On Mac, keys return codes like 1113938.  Masking with 0xFF limits it to 0-255.
    if key == Keys.UNKNOWN:
        print("Unknown cv2 Key Code: {}".format(cvkeycode))
    return key


def cv_window_input(  # Get text input from a cv2 window.
        prompt: str,  # A text prompt
        window_size: Optional[Tuple[int, int]] = None,  # Optionally, window size (otherwise it will just expand to fit)
        timeout=30,  # Timeout for user input (raise InputTimeoutError if no response in this time)
        return_none_if_timeout=True,  # Just return None if timeout
        text_color=BGRColors.WHITE,  # Text color
        background_color=BGRColors.DARK_GRAY,  # Background color
        window_name='User Input (Enter to complete, Exc to Cancel)'  # Name of CV2 windot
) -> Optional[str]:  # The Response, or None if you press ESC

    displayer = TextDisplayer(text_color=text_color, background_color=background_color, size=window_size)
    next_cap = False
    character_keys = {Keys.SPACE: '  ', Keys.PERIOD: '.>', Keys.COMMA: ',<', Keys.SEMICOLON: ';:', Keys.SLASH: '/?', Keys.DASH: '-=', Keys.EQUALS: '=+'}
    response = ''
    while True:
        img = displayer.render('{}\n >> {}'.format(prompt, response))
        cv2.imshow(window_name, img)
        key = cvkey_to_key(cv2.waitKey(int(timeout * 1000)))
        if key is None:
            if return_none_if_timeout:
                return None
            else:
                raise InputTimeoutError("User provided no input for {:.2f}s".format(timeout))
        elif key == Keys.RETURN:
            cv2.destroyWindow(window_name)
            return response
        elif key == Keys.ESC:
            cv2.destroyWindow(window_name)
            return None
        elif len(key) == 1:
            response += key.upper() if next_cap else key.lower()
            next_cap = False
        elif key in character_keys:
            base_key, shift_key = character_keys[key]
            response += shift_key if next_cap else base_key
            next_cap = False
        elif key == Keys.PERIOD:
            response += '.'
        elif key == Keys.PERIOD:
            response += '.'
        elif key == Keys.BACKSPACE:
            response = response[:-1]
        elif key in (Keys.LSHIFT, Keys.RSHIFT):
            next_cap = True
        else:
            print("Don't know how to handle key '{}'.  Skipping.".format(key))


def put_text_at(img, text, pos=(0, -1), scale=1, color=(0, 0, 0), thickness=1, font=cv2.FONT_HERSHEY_PLAIN, dry_run=False):
    """
    Add text to an image
    :param img:  add to this image
    :param text: add this text
    :param pos:  (x, y) location of text to add, pixel values, point indicates bottom-left corner of text.
    :param scale: size of text to add
    :param color:  (r,g,b) uint8
    :param thickness:  for adding text
    :param font:  font constant from cv2
    :param dry_run:  don't add to text, just calculate size required to add text
    :return:  dict with 'x': [x_min, x_max], 'y': [y_min, y_max], 'baseline': location of text baseline relative to y_max
    """
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    y_pos = pos[1] + baseline if pos[1] >= 0 else img.shape[0] + pos[1] + baseline
    x_pos = pos[0] if pos[0] >= 0 else img.shape[1] + pos[0]
    box = {'y': [y_pos - size[1], y_pos],
           'x': [x_pos, x_pos + size[0]],
           'baseline': baseline}
    if not dry_run:
        cv2.putText(img, text, (x_pos, y_pos), font, scale, color, thickness, bottomLeftOrigin=False)
    return box


def create_gap_image(  # Generate a colour image filled with one colour
        size: Tuple[int, int],  # Image (width, height))
        gap_colour: Optional[Tuple[int, int, int]] = None  # BGR color to fill gap, or None to use default
) -> 'array(H,W,3)[uint8]':
    if gap_colour is None:
        gap_colour = DEFAULT_GAP_COLOR

    width, height = size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img += np.array(gap_colour, dtype=np.uint8)
    return img


def resize_to_fit_in_box(img: BGRImageArray, size: Union[int, Tuple[int, int]], expand=True, shrink=True, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image to fit in a box
    :param img: Imageeeee
    :param size: Box (width, height) in pixels
    :param expand: Expand to fit
    :param shrink: Shrink to fit
    :param interpolation: cv2 Interpolation enum, e.g. cv2.INTERP_NEAREST
    :return: The resized images
    """
    if isinstance(size, (int, float)):
        size = (size, size)
    ratio = min(float(size[0]) / img.shape[1] if size[0] is not None else float('inf'), float(size[1]) / img.shape[0] if size[1] is not None else float('inf'))
    if (shrink and ratio < 1) or (expand and ratio > 1):
        img = cv2.resize(img, fx=ratio, fy=ratio, dsize=None, interpolation=interpolation)
    return img


def draw_image_to_region_inplace(  # Assign an image to a region in a parent image
        parent_image,  # type: array(WP,HP,...)[uint8]  # The parent image into which to draw inplace
        img,  # type: array(W,H,...)[uint8]  # The image to draw in the given region
        xxyy_region=None,  # type: Optional[Tuple[int, int, int, int]]  # The (left, right, bottom, top) edges (right/top-non-inclusive) to drawbottom,
        expand=True,  # True to expand image to fill region
        gap_colour=None,  # type: Optional[Tuple[int, int, int]]  # BGR fill colour
        fill_gap=True  # True to fill in the gaps at the edges with fill colour
):
    if gap_colour is None:
        gap_colour = DEFAULT_GAP_COLOR

    x1, x2, y1, y2 = xxyy_region if xxyy_region is not None else (0, parent_image.shape[1], 0, parent_image.shape[0])
    width, height = x2 - x1, y2 - y1
    resized = resize_to_fit_in_box(img, size=(width, height), expand=expand, shrink=True)
    xs, ys = (x1 + (width - resized.shape[1]) // 2), (y1 + (height - resized.shape[0]) // 2)
    if fill_gap:
        parent_image[y1:y2, x1:x2] = gap_colour
    parent_image[ys:ys + resized.shape[0], xs:xs + resized.shape[1]] = resized


def draw_multiple_images_inplace(  # Draw multiple images into a parent image
        parent_image,  # type: 'array(HP,WP,3)[uint8]'  # The image in which to draw
        image_dict,  # type: Dict[str, 'array(H,W,3)[uint8]']  # The images to insert
        xxyy_dict,  # type: Dict[str, Tuple[int, int, int, int]]  # The bounding boxes (referenced by the same keys as image_dict)
        float_norm='max',  # type: str  # How to map floats to color range.  See to_uint8_color_image
        expand=True,  # True to expand image to fit bounding box
        gap_colour=None,  # type: Optional[Tuple[int, int, int]]  # The Default BGR color to fill the gaps
):
    for name, img in image_dict.items():
        assert name in xxyy_dict, "There was no bounding box for image named '{}'".format(name)
        draw_image_to_region_inplace(parent_image=parent_image, img=img, xxyy_region=xxyy_dict[name], expand=expand, gap_colour=gap_colour, fill_gap=False)


@attrs
class WindowLayout(object):
    """ Object defining the location of named boxes within a window. """
    panel_xxyy_boxes = attrib(type=Dict[str, Tuple[int, int, int, int]])
    size = attrib(type=Tuple[int, int], default=None)

    def __attrs_post_init__(self):
        if self.size is None:
            self.size = (max(x for _, x, _, _ in self.panel_xxyy_boxes.values()), max(y for _, _, _, y in self.panel_xxyy_boxes.values()))

    def render(self,  # Given a dictionary of images, render it into a single image
               image_dict,  # type: Dict[str, 'array(H,W,3)[uint8']
               gap_color=None,  # type: Optional[Tuple[int, int, int]]  # The Default BGR color to fill the gaps
               float_norm='max'  # If images are passed as floats, how to normalize them (see to_uint8_color_image)
               ):  # type: (...) -> 'array(H,W,3)[uint8]'  # The rendered image

        parent_frame = create_gap_image(self.size, gap_colour=gap_color)
        draw_multiple_images_inplace(parent_image=parent_frame, image_dict=image_dict, xxyy_dict=self.panel_xxyy_boxes, float_norm=float_norm, expand=False)
        return parent_frame


class RowColumnPacker:
    """ Use to pack boxes by defining nested rows and columns (use Row/Col subclasses for convenience)

        packer = Row(Col(Row('panel_1', 'panel_2'),
                         'panel_3'),
                     'panel_4')
        layout = packer.pack_boxes({'panel_1': (300, 400), 'panel_2': (200, 200), 'panel_3': (600, 300), 'panel_4': (700, 700)})

    """

    OTHER = None  # Used to indicate that

    def __init__(self, *args: Union['RowColumnPacker', str, None], orientation='h'):
        """
        :param args: The panels in this object.  This can either be a nested Row/Col/RowColumnPacker object, or a string panel name, or RowColumnPacker.OTHER to indicate
            that this panel takes any unclaimed windows.
        :param orientation: How to stack them: 'h' for horizontal or 'v' for vertical
        """
        assert orientation in ('h', 'v')
        assert all(isinstance(obj, Hashable) for obj in args), f"Unhashable elements in args: {args}"
        self.orientation = orientation
        self.items = args

    def pack_boxes(self,  # Pack images into a resulting images
                   box_size_dict: Dict[str, Tuple[int, int]],
                   ) -> WindowLayout:

        new_contents_dict = {}

        # We reorder so that the window with the "OTHER" box is packed last (after all images with panels have been assigned)
        reordered_items = sorted(self.items, key=lambda item_: isinstance(item_, RowColumnPacker) and RowColumnPacker.OTHER in item_.get_all_items())

        # Get the layout for each sub-widow
        window_layouts: List[WindowLayout] = []
        remaining_boxes = box_size_dict
        for item in reordered_items:
            if isinstance(item, RowColumnPacker):
                window_layouts.append(item.pack_boxes(remaining_boxes))
            elif item in box_size_dict:
                window_layouts.append(WindowLayout(size=box_size_dict[item], panel_xxyy_boxes={item: (0, box_size_dict[item][0], 0, box_size_dict[item][1])}))
            elif item is RowColumnPacker.OTHER:
                child_frame = RowColumnPacker(*remaining_boxes.keys(), orientation=self.orientation)
                window_layouts.append(child_frame.pack_boxes(remaining_boxes))
            else:  # This panel is not included in the data, which is ok, we just skip.
                continue
            remaining_boxes = {name: box for name, box in remaining_boxes.items() if name not in window_layouts[-1].panel_xxyy_boxes}

        # Combine them into a single window layout
        if self.orientation == 'h':
            total_height = max([0] + [layout.size[1] for layout in window_layouts])
            total_width = 0
            for layout in window_layouts:
                width, height = layout.size
                v_offset = (total_height - height) // 2
                for name, (xmin, xmax, ymin, ymax) in layout.panel_xxyy_boxes.items():
                    new_contents_dict[name] = (xmin + total_width, xmax + total_width, ymin + v_offset, ymax + v_offset)
                total_width += width
        else:
            total_width = max([0] + [layout.size[0] for layout in window_layouts])
            total_height = 0
            for layout in window_layouts:
                width, height = layout.size
                h_offset = (total_width - width) // 2
                for name, (xmin, xmax, ymin, ymax) in layout.panel_xxyy_boxes.items():
                    new_contents_dict[name] = [xmin + h_offset, xmax + h_offset, ymin + total_height, ymax + total_height]
                total_height += height

        return WindowLayout(panel_xxyy_boxes=new_contents_dict, size=(total_width, total_height))

    def get_all_items(self) -> Set[str]:
        return {name for item in self.items for name in
                (item.get_all_items() if isinstance(item, RowColumnPacker) else [item])}  # pylint:disable=superfluous-parens

    def concat(self, *objects):
        return RowColumnPacker(*([self.items] + list(objects)), orientation=self.orientation)

    def __len__(self):
        return len(self.items)


class Row(RowColumnPacker):
    """ A row of panels """

    def __init__(self, *args):
        RowColumnPacker.__init__(self, *args, orientation='h')


class Col(RowColumnPacker):
    """ A column of panels """

    def __init__(self, *args):
        RowColumnPacker.__init__(self, *args, orientation='v')


@attrs
class EasyWindow(object):
    """ Contains multiple updating subplots """
    box_packer = attrib(type=RowColumnPacker)
    identifier = attrib(default=DEFAULT_WINDOW_NAME)
    panel_scales = attrib(factory=dict, type=Dict[Optional[str], float])  # e.g. {'panel_name': 2.}
    skip_title_for = attrib(factory=set, type=Set[Optional[str]])  # e.g. {'panel_name'}
    images = attrib(factory=OrderedDict)
    gap_color = attrib(default=DEFAULT_GAP_COLOR)
    title_background = attrib(default=np.array([50, 50, 50], dtype=np.uint8))
    _last_size_and_layout = attrib(type=Optional[Tuple[Dict[str, Tuple[int, int]], WindowLayout]], default=None)

    ALL_SUBPLOTS = None  # Flag that can be used in panel_scales and skip_title_for to indicate that all subplots should have this property

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self,  # Update the window and maybe show it.
               image: BGRImageArray,  # The data to plot (if display not provided, we try to infer appropriate plot type from data)
               name: str,  # The name of the "plot panel" to plot it in
               scale: Optional[float] = None,  # Optionally, how much to scale the image
               skip_none: bool = False,  # If data is None, just skip the update
               add_title: Optional[bool] = None,  # Add a totle matching the name,
               title: Optional[str] = None,  # Optional title to put at top instead of name
               ):
        if image is None:
            del self.images[name]
            return

        # Allow panel settings to be set
        add_title = add_title if add_title is not None else (name not in self.skip_title_for and EasyWindow.ALL_SUBPLOTS not in self.skip_title_for)
        scale = scale if scale is not None else self.panel_scales[name] if name in self.panel_scales else self.panel_scales.get(EasyWindow.ALL_SUBPLOTS, None)
        if scale is not None:
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if add_title:
            title_sec = create_gap_image((image.shape[1], 30), gap_colour=self.title_background)
            put_text_at(img=title_sec, pos=(0, 15), text=title or name, color=(255, 255, 255))
            image = np.vstack([title_sec, image])
        if image is None and skip_none:
            return
        self.images[name] = image

    def render(self) -> BGRImageArray:
        """ Render the image into an array """
        sizes = OrderedDict((name, (im.shape[1], im.shape[0])) for name, im in self.images.items())
        if self._last_size_and_layout is not None and self._last_size_and_layout[0] == sizes:
            _, window_layout = self._last_size_and_layout
        else:
            window_layout = self.box_packer.pack_boxes(sizes)  # type: WindowLayout
            self._last_size_and_layout = (sizes, window_layout)
        return window_layout.render(image_dict=self.images, gap_color=self.gap_color)

    def close(self):  # Close this plot window
        try:
            cv2.destroyWindow(self.identifier)
        except cv2.error:
            pass  # It's ok, it's already gone
            print("/\\ Ignore above error, it's fine")  # Because cv2 still prints it
