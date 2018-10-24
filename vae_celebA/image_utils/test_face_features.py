from artemis.general.test_mode import TestMode
from quva_code.utils.face_features import get_lab_face_vgg_features_dataset
import numpy as np
from quva_code.utils.lab_face_dataset import get_lab_face_labels

__author__ = 'peter'


def test_get_vgg_features_dataset_from_paths():
    with TestMode(False):  # Memoization is normally automatically disabled in tests, but we want to allow it for this test to save time.
        training_features, training_labels, test_features, test_labels = get_lab_face_vgg_features_dataset(clear_caches=False, output_layer = 'fc7')
        assert training_features.ndim==2 and test_features.ndim==2
        assert training_features.shape[1] == 4096 and test_features.shape[1] == 4096
        assert training_features.shape[0] == len(training_labels)
        assert np.allclose((training_features**2).sum(axis=1), 1)
        assert np.allclose((test_features**2).sum(axis=1), 1)
        assert test_features.shape[0] == len(test_labels)
        assert training_features.shape[0] > test_features.shape[0]  # (Because we allot at least 50% for each class to training)
        assert all(test_name in training_labels for test_name in test_labels)


def test_labels_order_right():
    _, training_indices, _, test_indices = get_lab_face_vgg_features_dataset(clear_caches=False, output_layer = None, numeric_labels = True)
    _, training_labels, _, test_labels = get_lab_face_vgg_features_dataset(clear_caches=False, output_layer = None, numeric_labels = False)
    labels = get_lab_face_labels()
    assert tuple(labels[ix] for ix in training_indices)==training_labels
    assert tuple(labels[ix] for ix in test_indices)==test_labels


if __name__ == '__main__':
    test_labels_order_right()
