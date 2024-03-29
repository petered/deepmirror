
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def encoder(input_imgs, is_train = True, reuse = False, z_dim = 100):
    '''
    input_imgs: the input images to be encoded into a vector as latent representation. size here is [b_size,64,64,3]
    '''
    # z_dim = FLAGS.z_dim # 100
    ef_dim = 32 # encoder filter number

    # with tf.device("/cpu:0"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("encoder", reuse = reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_imgs, name='en/in') # (b_size,64,64,3)
        net_h0 = Conv2d(net_in, ef_dim, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='en/h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='en/h0/batch_norm')
        # net_h0.outputs._shape = (b_size,32,32,32)

        net_h1 = Conv2d(net_h0, ef_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='en/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='en/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,16,16,64)

        net_h2 = Conv2d(net_h1, ef_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='en/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='en/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,8,8,128)

        net_h3 = Conv2d(net_h2, ef_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='en/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='en/h3/batch_norm')
        # net_h2.outputs._shape = (b_size,4,4,256)

        # mean of z
        net_h4 = FlattenLayer(net_h3, name='en/h4/flatten')
        # net_h4.outputs._shape = (b_size,4*4*256)
        net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.identity,
                W_init = w_init, name='en/out1/lin_sigmoid')
        # net_out1 = BatchNormLayer(net_out1, act=tf.identity,
        #         is_train=is_train, gamma_init=gamma_init, name='en/out1/batch_norm')

        # net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.nn.relu,
        #         W_init = w_init, name='en/h4/lin_sigmoid')
        z_mean = net_out1.outputs # (b_size,100)

        # log of variance of z(covariance matrix is diagonal)
        net_h5 = FlattenLayer(net_h3, name='en/h5/flatten')
        net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.identity,
                W_init = w_init, name='en/out2/lin_sigmoid')
        # net_out2 = BatchNormLayer(net_out2, act=tf.nn.softplus,
        #         is_train=is_train, gamma_init=gamma_init, name='en/out2/batch_norm')
        # net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.nn.relu,
        #         W_init = w_init, name='en/h5/lin_sigmoid')
        z_log_sigma_sq = net_out2.outputs + 1e-6# (b_size,100)

    return net_out1, net_out2, z_mean, z_log_sigma_sq

def generator(inputs, image_size, c_dim, batch_size, is_train = True, reuse = False):
    '''
    generator of GAN, which can also be seen as a decoder of VAE
    inputs: latent representation from encoder. [b_size,z_dim]
    '''
    # image_size = FLAGS.output_size # 64 the output size of generator
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16) # 32,16,8,4
    gf_dim = 32
    # c_dim = FLAGS.c_dim # n_color 3f
    # batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse = reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        # net_h0.outputs._shape = (b_size,256*4*4)
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
        # net_h0.outputs._shape = (b_size,4,4,256)
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        # upsampling
        net_h1 = UpSampling2dLayer(net_h0, size=[8, 8], is_scale=False, method=1, 
                                    align_corners=False, name='g/h1/upsample2d')
        net_h1 = Conv2d(net_h1, gf_dim*4, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h1/conv2d')
        # net_h1 = DeConv2d(net_h0, gf_dim*4, (3, 3), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,8,8,128)

        net_h2 = UpSampling2dLayer(net_h1, size=[16, 16], is_scale=False, method=1, 
                                    align_corners=False, name='g/h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim*2, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h2/conv2d')
        # net_h2 = DeConv2d(net_h1, gf_dim*2, (3, 3), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,16,16,64)

        net_h3 = UpSampling2dLayer(net_h2, size=[32, 32], is_scale=False, method=1, 
                                    align_corners=False, name='g/h3/upsample2d')
        net_h3 = Conv2d(net_h3, gf_dim, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h3/conv2d')
        # net_h3 = DeConv2d(net_h2, gf_dim//2, (3, 3), out_size=(image_size, image_size), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')
        # net_h3.outputs._shape = (b_size,32,32,32)

        # no BN on last deconv
        # net_h4 = DeConv2d(net_h3, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4 = UpSampling2dLayer(net_h3, size=[64, 64], is_scale=False, method=1, 
                                    align_corners=False, name='g/h4/upsample2d')
        net_h4 = Conv2d(net_h4, c_dim, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h4/conv2d')
        # net_h4.outputs._shape = (b_size,64,64,3)
        # net_h4 = Conv2d(net_h3, c_dim, (5,5),(1,1), padding='SAME', W_init=w_init, name='g/h4/conv2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

