from artemis.fileman.disk_memoize import memoize_to_disk
from artemis.general.should_be_builtins import memoize, bad_value
import numpy as np
from plato.tools.pretrained_networks.vggnet import im2vgginput
from quva_code.distributed_jewel.utils.vgg16 import get_vgg_output, preprocess_for_vggnet
from quva_code.jewel.demo_lab_face_identifier import FaceFeatureExtractorTheano
from quva_code.utils.face_detection import FaceDetectorExtractor, get_face_detector_version
from quva_code.utils.lab_face_dataset import get_lab_face_dataset_from_paths, get_faces_image_locs, labels_to_indices

__author__ = 'peter'


class FaceFeatureExtractorTF(object):
    """
    The TensorFlow version of the VGGFace feature extractor.

    Eats faces and shits features.
    """

    def __init__(self, output_layer = 'fc7', n_workers = 1):
        import tensorflow as tf

        print ('Loading VGGFace Network...')
        self.input_pl = tf.placeholder(dtype=tf.float32,shape=(None, 224, 224, 3))
        # input_pl = tf.constant(vgg_input, dtype=tf.float32)
        from quva_code.distributed_jewel.utils.vgg16 import get_vgg_output
        vgg_out = get_vgg_output(self.input_pl, output_layer = output_layer, after_activation=False)
        self.vgg_out_norm = vgg_out/tf.sqrt(tf.reduce_sum(vgg_out**2,reduction_indices=1, keep_dims=True))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75*(1./n_workers))
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False,log_device_placement=False)
        self.session = tf.Session(config=sess_config)
        print ('Done.')

    def __call__(self, images):
        """
        A list of images: (n_samples, )(size_y, size_x, 3)
        An array of features: (n_samples, n_features)
        """
        prepped_image_data = np.array([preprocess_for_vggnet(im) if im.ndim==3 else preprocess_for_vggnet(np.repeat(im[:, :, None], 3, axis=2)) for im in images])  # (n_samples, 224, 224, 3)
        features = self.session.run(self.vgg_out_norm, feed_dict={self.input_pl: prepped_image_data})  # (n_samples, n_features)
        return features

    @staticmethod
    @memoize
    def get_singleton(**kwargs):
        """
        :return: A FaceFeatureExtractor

        Note that due to the memoize keyword, it will return the same instance of this feature extractor each time it is
        called with the same arguments (useful to avoid re-loading vggnet every time)
        """
        return FaceFeatureExtractorTF(**kwargs)


@memoize_to_disk
def get_vgg_features_from_images(images, output_layer = 'fc7'):
    """
    :param images: A list of images
    :return: An array of features
    """
    feature_extractor = FaceFeatureExtractorTF.get_singleton(output_layer=output_layer)
    features = feature_extractor(images)
    return features


@memoize_to_disk
def get_vgg_features_dataset_from_paths(path_name_pairs, face_detector, split_ratio = 0.5, max_image_loading_resolution = (400, 400),
            print_n_faces_found = False, output_layer = 'fc7', normalize_features = True, numeric_labels = False,
            feature_extractor_version = 'tensorflow', n_workers=1):
    """

    :param path_name_pairs: A list of (local_image_path, label) pairs
    :param face_detector: The FaceExtractor object to use (see how 'default' is instantiated below for an example)
    :param split_ratio: A float in (0, 1), that determines the minimum fraction of samples of a given class to be
        allotted to the training set.
    :param max_image_loading_resolution: The max resolution to load the images at (affects the speed of face detection).
    :param print_n_faces_found: True if you want to print the number of faces found in each image.
    :param output_layer: A string identifying the output feature layer (eg 'fc7')
        If set to "None" the
    :param normalize_features: L2-Normalize the output features
    :param numeric_labels: Return integer labels instead of strings.
    :param n_workers: Number of workers to use for the TensorFlow VGGNet implementation
    :return: training_features, training_labels, test_features, test_labels:
        training_features is a (n_training_samples, n_features) array of features (or (n_training_samples, size_y, size_x, 3) if output_layer is None)
        training_labels is a length:n_training_samples list of strings identifying the name associated with the feature
        test_features is a (n_test_samples, n_features) array of features (or (n_test_samples, size_y, size_x, 3) if output_layer is None)
        test_labels is a length:n_test_samples list of strings identifying the name associated with the feature
    """

    assert feature_extractor_version in ('theano', 'tensorflow')
    training_face_name_pairs, test_face_name_pairs = get_lab_face_dataset_from_paths(
        path_name_pairs = path_name_pairs,
        face_detector = face_detector,
        split_ratio = split_ratio,
        max_resolution = max_image_loading_resolution,
        print_n_faces_found = print_n_faces_found,
        )
    training_faces, training_labels = zip(*training_face_name_pairs)
    test_faces, test_labels = zip(*test_face_name_pairs)

    if output_layer is None:
        if feature_extractor_version == 'theano':  # Preprocess in the manner that the theano net expects.
            training_features = im2vgginput(training_faces, already_bgr=False)
            test_features = im2vgginput(test_faces, already_bgr=False)
        else:
            training_features = np.array([preprocess_for_vggnet(im) for im in training_faces])
            test_features = np.array([preprocess_for_vggnet(im) for im in test_faces])
    else:
        if feature_extractor_version == 'theano':
            feature_extractor = FaceFeatureExtractorTheano(output_layer=output_layer, normalize_outputs=normalize_features)
        elif feature_extractor_version == 'tensorflow':
            assert normalize_features, "Have not yet set up the non-normed version of the tensorflow feature extractor."
            feature_extractor = FaceFeatureExtractorTF(output_layer = output_layer, n_workers=n_workers)
        else:
            bad_value(feature_extractor_version)
        print 'Extracting face Features...'
        training_features = feature_extractor(training_faces)
        test_features = feature_extractor(test_faces)
        print 'Done.'

    if numeric_labels:
        training_names = training_labels
        training_labels = labels_to_indices(training_labels, training_names)  # (n_training_samples, ) integer array of indices
        test_labels = labels_to_indices(test_labels, training_names)  # (n_training_samples, ) integer array of indices

    return training_features, training_labels, test_features, test_labels


@memoize
def get_lab_face_vgg_features_dataset(clear_caches = False, max_samples = None, subdirectory = 'selfies',
        face_detector_version = 'default', **feature_extraction_kwargs):
    """
    Get the features dataset.

    :param clear_caches: True to delete and re-download the dataset, clear the cache of the memoized face/feature extraction function.
    :param max_samples:
    :param subdirectory:
    :param face_detector_version:
    :param path_name_pairs: A list of (local_image_path, label) pairs
    :param face_detector: The FaceExtractor object to use (see how 'default' is instantiated below for an example)
    :param split_ratio: A float in (0, 1), that determines the minimum fraction of samples of a given class to be
        allotted to the training set.
    :param max_image_loading_resolution: The max resolution to load the images at (affects the speed of face detection).
    :param print_n_faces_found: True if you want to print the number of faces found in each image.
    :param output_layer: A string identifying the output feature layer (eg 'fc7')
        If set to "None" the
    :param n_workers: Number of workers to use for the TensorFlow VGGNet implementation
    :return: training_features, training_labels, test_features, test_labels:
        training_features is a (n_training_samples, n_features) array of features (or (n_training_samples, size_y, size_x, 3) if output_layer is None)
        training_labels is a length:n_training_samples list of strings identifying the name associated with the feature
        test_features is a (n_test_samples, n_features) array of features (or (n_test_samples, size_y, size_x, 3) if output_layer is None)
        test_labels is a length:n_test_samples list of strings identifying the name associated with the feature
    """

    path_name_pairs = get_faces_image_locs(force_download = clear_caches, max_samples = max_samples, subdirectory = subdirectory)
    face_detector = get_face_detector_version(face_detector_version)
    if clear_caches:
        get_vgg_features_dataset_from_paths.clear_cache()
    return get_vgg_features_dataset_from_paths(path_name_pairs=path_name_pairs, face_detector=face_detector,
            **feature_extraction_kwargs)


def get_lab_face_vgg_features_dataset_version(version_name, clear_caches = False):
    """
    A shorthand for loading various versions of the dataset.
    :param version_name:
    :param clear_caches:
    :return:
    """
    kwargs = {
        'default': dict(),
        'faces': dict(output_layer = None),
        'theano_features_numlabels': dict(output_layer = 'fc7', face_detector_version='default', split_ratio = 0.5, feature_extractor_version = 'theano', numeric_labels = True),
        'theano_faces_numlabels': dict(output_layer = None, face_detector_version='default', split_ratio = 0.5, feature_extractor_version = 'theano', numeric_labels = True),
        'theano_faces': dict(output_layer = None, face_detector_version='default', split_ratio = 0.5, feature_extractor_version = 'theano')
        }[version_name]
    return get_lab_face_vgg_features_dataset(clear_caches = clear_caches, **kwargs)
