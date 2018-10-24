from collections import OrderedDict
from artemis.fileman.disk_memoize import memoize_to_disk
from plato.tools.convnet.conv_specifiers import ConvolverSpec, NonlinearitySpec, DropoutSpec
from plato.tools.convnet.convnet import ConvNet
from quva_code.jewel.face_rec import load_vgg_face_spec
from quva_code.utils.face_features import get_lab_face_vgg_features_dataset
import numpy as np
from quva_code.utils.predictors import MultinomialRegression
from spiking_experiments.spiking_convnet.normalized_vggnet import normalize_convnet_network_activations
from utils.benchmarks.predictor_comparison import assess_online_predictor
from utils.datasets.datasets import DataSet

__author__ = 'peter'


def get_trained_vggface_net(clear_caches=False, **kwargs):
    if clear_caches:
        get_trained_vggface_specs.clear_cache()
    layer_specs = get_trained_vggface_specs(**kwargs)
    net = ConvNet.from_init(layer_specs, input_shape = (3, 224, 224))
    return net


@memoize_to_disk
def get_trained_vggface_specs(
        feature_layer = 'relu7',
        clear_caches = False,
        remove_dropout = True,
        normalize_activations = False,
        normalization_desired_scale=1,
        normalization_desired_input_scale=1,
        rng = 1234,
        learning_rate=0.0001,
        n_epochs=10,
        minibatch_size='full'
        ):
    """

    :param feature_layer:
    :param normalize_activations:
    :param normalization_desired_scale:
    :param normalization_desired_input_scale:
    :param training_args:
    :return:
    """

    training_faces, training_labels, test_faces, test_labels = get_lab_face_vgg_features_dataset(
        clear_caches = clear_caches, output_layer = None, face_detector_version='default', feature_extractor_version='theano',
        normalize_features = False, numeric_labels = True
        )

    layer_specs = load_vgg_face_spec(up_to_layer=feature_layer)

    if remove_dropout:
        layer_specs = remove_dropout_layers(layer_specs)

    f = ConvNet.from_init(layer_specs, input_shape = training_faces.shape[1:], rng=rng).test_call.compile(add_test_values = False)

    training_features = f(training_faces).reshape(training_faces.shape[0], -1)
    test_features = f(test_faces).reshape(test_faces.shape[0], -1)

    w = get_final_regression_weights(
        training_features=training_features,
        training_labels=training_labels,
        test_features=test_features,
        test_labels=test_labels,
        learning_rate=learning_rate,
        n_epochs = n_epochs,
        minibatch_size = minibatch_size,
        )

    layer_specs['fc_lab']=ConvolverSpec(w=w.T[:, :, None, None], b=False, mode='valid')

    if normalize_activations:
        new_layer_specs = normalize_convnet_network_activations(input_data = training_faces, layer_specs=layer_specs,
            desired_scale=normalization_desired_scale, desired_input_scale=normalization_desired_input_scale, leave_last_layer=False)
        return new_layer_specs
    else:
        return layer_specs


def get_final_regression_weights(training_features, training_labels, test_features, test_labels, learning_rate=0.0001,
            n_epochs=10, minibatch_size='full'):
    """

    :param training_features: A (n_training_samples, n_dims) array
    :param training_labels: A (n_training_samples, ) array of integer labels
    :param test_features: A (n_test_samples, n_dims) array
    :param test_labels: A (n_test_samples, ) array of integer labels
    :param learning_rate:
    :param n_epochs:
    :param minibatch_size:
    :return: A (n_dims, max(training_labels)+1) arraw representing the weight matrix learned from regression.
    """
    dataset = DataSet.from_xyxy(
        training_inputs = training_features,
        training_targets = training_labels,
        test_inputs = test_features,
        test_targets = test_labels,
        )
    predictor = MultinomialRegression(n_in=training_features.shape[1], n_out=np.max(training_labels) + 1, learning_rate=learning_rate)
    _ = assess_online_predictor(
        predictor=predictor,
        dataset = dataset,
        minibatch_size=minibatch_size,
        evaluation_function='percent_correct',
        test_epochs=np.arange(n_epochs+1),
        )
    return predictor.w


def remove_dropout_layers(layers_specs):
    """
    Remove dropout layers from the specs without changing the test-time function of the
    network, by multiplying downstream weights by the dropout factor.
    :param layers_specs:
    :return:
    """

    layer_spec_iter = layers_specs.iteritems()
    new_specs = OrderedDict()
    for spec_name, spec in layer_spec_iter:
        # spec = spec_iter.next()
        if isinstance(spec, DropoutSpec):
            spec_name, next_spec = layer_spec_iter.next()
            assert isinstance(next_spec, ConvolverSpec)
            new_convolver = next_spec.clone()
            new_convolver.w = new_convolver.w / spec.dropout_rate
            new_specs[spec_name] = next_spec
        else:
            new_specs[spec_name] = spec
    return new_specs
