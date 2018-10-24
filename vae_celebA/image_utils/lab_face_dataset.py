from artemis.fileman.disk_memoize import memoize_to_disk
from artemis.fileman.file_getter import get_archive
from artemis.fileman.smart_io import smart_load_image
import os
import numpy as np


__author__ = 'peter'


def get_selfies_folder(subdirectory = 'selfies', force_download = False):
    jewel_data_path = get_archive(relative_path='data/jewel_data', url='https://github.com/QUVA-Lab/jewel_data/archive/master.zip', force_download=force_download)
    selfies_folder_path = os.path.join(jewel_data_path, 'jewel_data-master', subdirectory)
    return selfies_folder_path


def get_faces_image_locs(force_download = False, max_samples = None, subdirectory = 'selfies'):
    """
    :return: A list of 2-tuples of (local_image_path, name_of_person),
    """
    selfies_folder_path = get_selfies_folder(subdirectory, force_download=force_download)
    assert os.path.isdir(selfies_folder_path), "Somethin' ain't riaaght."
    path_name_pairs = []
    for roots, dir, files in os.walk(selfies_folder_path):
        if len(files) > 0 and (files[0].endswith(".jpg") or files[0].endswith(".JPG")):
            name = roots.split("/")[-1]
            for f in files:
                if len(path_name_pairs)==max_samples:
                    break
                path_name_pairs.append((os.path.join(roots, f), name))
    return path_name_pairs


def get_lab_pictures(force_download = False, force_rgb = True, max_resolution = (400, 400)):
    """
    :param force_download: Set to true to delete the local cache and re-download (useful if the online set of images has
        changed).
    :return: A list of colour OR BW images (n_samples, )((size_y, size_x, 3) OR (size_y, size_x))
    """
    locs=get_faces_image_locs(force_download=force_download)
    path_name_pairs = locs
    print 'Loading Lab Pictures...'
    images = [smart_load_image(path, max_resolution=max_resolution, force_rgb=force_rgb) for path, _ in path_name_pairs]
    print 'Done.'
    return images


def get_lab_faces_dataset(force_download = False, **kwargs):
    """
    :param force_download: Remove the cached dataset and forcibly re-download the files.
    :param kwargs: See function get_lab_face_dataset_from_paths below for additional arguments
    :return: See function get_lab_face_dataset_from_paths below for what is returned
    """
    path_name_pairs = get_faces_image_locs(force_download=force_download)
    return get_lab_face_dataset_from_paths(path_name_pairs=path_name_pairs, **kwargs )


def get_lab_face_dataset_from_paths(path_name_pairs, face_detector = 'default', max_resolution = (400, 400),
        print_n_faces_found = False, split_ratio = 0.5):
    """
    :param path_name_pairs: A list of (local_image_path, label) pairs
    :param face_detector: The FaceExtractor object to use (see how 'default' is instantiated below for an example)
    :param max_resolution: Max road:
    :param print_n_faces_found:
    : (training_face_name_pairs, test_face_name_pairs)   Where..
        training_face_name_pairs is a list of 2-tuples of (face_image_array, name_of_person)
        test_face_name_pairs is a list of 2-tuples of (face_image_array, name_of_person)

    Images are split such that:
    - There is no overlap between the training and test set
    - There are more images per class in the training set than the test set (this may mean some people have no images in the test set)return:

    print 'Getting and extracting lab faces..'
    """
    if isinstance(face_detector, str):
        from quva_code.utils.face_detection import get_face_detector_version
        face_detector = get_face_detector_version(face_detector)

    image_label_pairs = [(smart_load_image(path, max_resolution = max_resolution, force_rgb=True), name) for path, name in path_name_pairs]
    faces_label_pairs = [(face_detector(image), name) for image, name in image_label_pairs]
    print 'Done.'
    if print_n_faces_found:
        print 'Number of faces found:\n  ' + '\n  '.join('%s: %s' % (path, len(faces)) for (path, _), (faces, _) in zip(path_name_pairs, faces_label_pairs))
    face_label_pairs = [(faces[np.argmax([f.size for f in faces])], name) for faces, name in faces_label_pairs if len(faces) > 0]
    return split_pairs_into_training_test(face_label_pairs, split_ratio = split_ratio)


def split_pairs_into_training_test(pairs, split_ratio = 0.5):
    """
    :param pairs: A list of pairs of (data_object, label)
    :return: training_pairs, test_pairs
        Each of these is a list of the same type.
    """
    labels = set(name for _, name in pairs)
    name_objects_dict = {lab: [data_obj for data_obj, this_label in pairs if this_label == lab] for lab in labels}
    training_pairs = [(data_obj, name) for name, data_objects in name_objects_dict.iteritems() for data_obj in data_objects[:int(np.ceil(len(data_objects)*split_ratio))]]
    test_pairs = [(data_obj, name) for name, data_objects in name_objects_dict.iteritems() for data_obj in data_objects[int(np.ceil(len(data_objects)*split_ratio)):]]
    return training_pairs, test_pairs


def labels_to_indices(labels, labels_list):
    """
    :param labels: A list of string labels
    :param labels_list: A list of strings containing all possible labels.
    :return:
    """
    assert all(n in labels_list for n in labels), "All names must be in the names_list"
    unique_names = sorted(list(set(labels_list)))
    indices = np.array([unique_names.index(n) for n in labels])
    return indices


@memoize_to_disk
def get_lab_face_labels(subdirectory = 'selfies', clear_caches = False):
    """
    This is horrible because we load the whole dataset just to get the labels but whatthehell
    it works and we memoize it.

    :return: The list of labels for the lab face dataset.
    """
    from quva_code.utils.face_features import get_lab_face_vgg_features_dataset
    _, training_face_labels, _, _ = get_lab_face_vgg_features_dataset(clear_caches=clear_caches, output_layer = None, subdirectory=subdirectory)
    sorted_labels = sorted(list(set(training_face_labels)))  # The sorted labels
    return sorted_labels
