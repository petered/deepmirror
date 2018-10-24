from plato.tools.common.config import float_precision
from quva_code.utils.face_features import get_lab_face_vgg_features_dataset_version
from quva_code.utils.pretrained_vggface_specs import get_trained_vggface_net
import numpy as np
__author__ = 'peter'


def test_get_trained_vggface_specs(clear_caches = False):

    training_faces, training_labels, test_faces, test_labels = get_lab_face_vgg_features_dataset_version('theano_faces_numlabels', clear_caches = False)
    net = get_trained_vggface_net(clear_caches=clear_caches)
    f = net.test_call.compile(add_test_values = False)
    training_out = f(training_faces)
    test_out = f(test_faces)
    training_score = 100*np.mean(np.argmax(training_out.reshape(training_out.shape[0], -1), axis=1)==training_labels)
    test_score = 100*np.mean(np.argmax(test_out.reshape(test_out.shape[0], -1), axis=1)==test_labels)
    print 'Training score: %.3g%%' % (training_score, )
    print 'Test score: %.3g%%' % (test_score, )
    assert training_score == 100
    assert test_score == 100
    # TODO: Find out why test score is 100% when training the predictor but 93.3% here (maybe normalized outputs from fe? in training?)


def test_normalized_vggface(clear_caches = False):

    with float_precision('float32'):
        training_faces, training_labels, test_faces, test_labels = get_lab_face_vgg_features_dataset_version('theano_faces_numlabels', clear_caches = False)
        normed_training_faces = training_faces / np.mean(np.abs(training_faces))
        normed_test_faces = test_faces / np.mean(np.abs(training_faces))
        net = get_trained_vggface_net(normalize_activations=True, normalization_desired_scale=1., normalization_desired_input_scale=1., clear_caches=clear_caches)
        f = net.get_named_layer_activations.partial(test_call=True).compile(add_test_values = False)
        train_activations = f(normed_training_faces)
        test_activations = f(normed_test_faces)
        test_score = 100*np.mean(np.argmax(test_activations.values()[-1].reshape(test_activations.values()[-1].shape[0], -1), axis=1)==test_labels)
        print 'Test score: %.3g%%' % (test_score, )
        assert test_score == 100
        assert all(np.isclose(np.mean(np.abs(train_activations[layer])), 1) for layer in test_activations if layer.startswith('relu'))


if __name__ == '__main__':

    # test_get_trained_vggface_specs()

    test_normalized_vggface()
