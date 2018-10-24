from artemis.fileman.smart_io import smart_load
from artemis.general.image_ops import equalize_image_dims
from artemis.plotting.data_conversion import put_list_of_images_in_array, put_data_in_grid
import matplotlib.pyplot as plt


__author__ = 'peter'


"""
Demonstrates our how we can take a series of images and put puctures into a grid for plotting
"""


def demo_draw_series_of_faces():

    saved_face_locations_dict = {
        'arnold': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/04/arnold-smeulders.jpg',
        'cees': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/05/cees2016-bw-150x150.jpg',
        'jan': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/04/jan-bergstra.jpg',
        'jeff': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/04/jeff-gehlhaar.jpg',
        'stratis': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/04/efstratios-gavves.jpg',
        'mijir': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/04/mihir-jain.jpg',
        'mijung': 'https://ivi.fnwi.uva.nl/quva/wp-content/uploads/2016/05/mijung-park.jpg',
        }

    face_arrays = [smart_load(addr, use_cache=True) for addr in saved_face_locations_dict.values()]  # A list of (size_y, size_x, 3) colour or (size_y, size_x) B/W image arrays
    equalized_face_arrays = equalize_image_dims(face_arrays, x_dim='max')  # A list of image arrays scaled to roughly the same size (but may still have different aspect ratios)
    image_array = put_list_of_images_in_array(equalized_face_arrays)  # A (n_images, size_y, size_x, 3) array of images.
    image_grid = put_data_in_grid(image_array)  # A (size_y, size_x, 3) array containing the images tiled into a grid
    plt.imshow(image_grid)
    plt.show()


if __name__ == '__main__':
    demo_draw_series_of_faces()
