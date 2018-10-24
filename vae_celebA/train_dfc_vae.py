import pprint
from glob import glob

from artemis.experiments.experiment_record import get_current_record_dir
from artemis.fileman.file_getter import get_file
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.plotting.db_plotting import hold_dbplots, dbplot
from vae_celebA.dfc_vae import *
from vae_celebA.utils import *
from vae_celebA.utils import get_image
from vae_celebA.vgg_loss import *

pp = pprint.PrettyPrinter()


'''
Tensorlayer implementation of DFC-VAE

Taken from https://github.com/yzwxx/vae-celebA
by Xianxu Hou, Linlin Shen, Ke Sun, Guoping Qiu

Code modified by Peter O'Connor
'''


def main(
        epoch=50,
        learning_rate=0.0005,
        beta1 = 0.5,
        train_size=np.inf,
        # batch_size=64,
        batch_size=4,  # TODO: Revert
        image_size=148,
        output_size=64,
        c_dim=3,
        z_dim=100,
        sample_step=100,
        save_step=800,
        # save_step=1,
        test_number='dfc_vae3',
        checkpoint_dir='checkpoint',
        sample_dir='samples',
        is_crop=True,
        save_sample_images = False,
        ):

    tl.files.exists_or_mkdir(checkpoint_dir)
    tl.files.exists_or_mkdir(sample_dir)

    # with tf.device("/gpu:0"):
    with tf.device("/cpu:0"):  # TODO: Revert
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32,[batch_size, output_size,
            output_size, c_dim], name='real_images')

        # normal distribution for generator
        z_p = tf.random_normal(shape=(batch_size, z_dim), mean=0.0, stddev=1.0)
        # normal distribution for reparameterization trick
        eps = tf.random_normal(shape=(batch_size, z_dim), mean=0.0, stddev=1.0)
        lr_vae = tf.placeholder(tf.float32, shape=[])


        # ----------------------encoder----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False, z_dim=z_dim)

        # ----------------------decoder----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks
        gen0, gen0_logits = generator(z, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=True, reuse=False) # reconstruction

        # ----------------------vgg net--------------------------
        vgg1_input = tf.image.resize_images(input_imgs,[224,224])
        net_in_real = InputLayer(vgg1_input, name='input1')
        conv1,l1_r,l2_r,l3_r,_,_ = conv_layers_simple_api(net_in_real,reuse=False)
        vgg1 = fc_layers(conv1,reuse=False)

        vgg2_input = tf.image.resize_images(gen0.outputs,[224,224])
        net_in_fake = InputLayer(vgg2_input, name='input2')
        conv2,l1,l2,l3,_,_ = conv_layers_simple_api(net_in_fake,reuse=True)
        vgg2 = fc_layers(conv2,reuse=True)


        # ----------------------for samples----------------------
        gen2, gen2_logits = generator(z, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=False, reuse=True)
        gen3, gen3_logits = generator(z_p, c_dim=c_dim, batch_size=batch_size, image_size=output_size, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        ''''
        reconstruction loss:
        use the learned similarity measurement in l-th layer(feature space) of pretrained VGG-16
        '''

        SSE_loss = tf.reduce_mean(tf.reduce_sum(tf.square(gen0.outputs - input_imgs),[1,2,3]))
        print(SSE_loss.get_shape(),type(SSE_loss))

        # perceptual loss in feature space in VGG net
        p1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l1 - l1_r), [1,2,3]))
        p2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l2 - l2_r), [1,2,3]))
        p3_loss = tf.reduce_mean(tf.reduce_sum(tf.square(l3 - l3_r), [1,2,3]))
        p_loss = p1_loss + p2_loss + p3_loss
        '''
        KL divergence:
        we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
        then compute KL divergence between z and standard normal gaussian N(0,I) 
        '''
        # train_vae
        KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq),1))
        print(KL_loss.get_shape(),type(KL_loss))

        ### important points! ###
        style_content_weight = 3e-5 # you may need to tweak this weight for a different dataset
        VAE_loss = KL_loss + style_content_weight*p_loss

        e_vars = tl.layers.get_variables_with_name('encoder',True,True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        vae_vars = e_vars + g_vars

        print("-------encoder-------")
        net_out1.print_params(False)
        print("-------generator-------")
        gen0.print_params(False)

        # optimizers for updating encoder and generator
        vae_optim = tf.train.AdamOptimizer(lr_vae, beta1=beta1).minimize(VAE_loss, var_list=vae_vars)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    npz = np.load(get_file('models/vgg16_weights.npz', url='https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz'))

    params = []
    for val in sorted( npz.items() ):
        print("  Loading %s" % str(val[1].shape))
        params.append(val[1])
    tl.files.assign_params(sess, params, vgg1)
    tl.files.assign_params(sess, params, vgg2)

    # prepare file under checkpoint_dir
    model_dir = "dfc-vae3"
    save_dir = os.path.join(checkpoint_dir, model_dir) #'./checkpoint/vae_0808'
    tl.files.exists_or_mkdir(save_dir)
    samples_1 = sample_dir + "/" + test_number
    tl.files.exists_or_mkdir(samples_1)
    data_files = glob(os.path.join(get_artemis_data_path(), 'data', 'celeba', 'img_align_celeba', "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files) # for tl.iterate.minibatches
    assert len(data_files)>0, 'No data files!'
    ##========================= TRAIN MODELS ================================##
    iter_counter = 0

    training_start_time = time.time()
    # use all images in dataset in every epoch
    for epoch in range(epoch):
        print("[*] Dataset shuffled!")
        minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=batch_size, shuffle=True)
        idx = 0
        batch_idxs = min(len(data_files), train_size) // batch_size
        while True:
            try:
                batch_files,_ = next(minibatch)
                batch = [get_image(batch_file, image_size, is_crop=is_crop, resize_w=output_size, is_grayscale = 0) for batch_file in batch_files if batch_file.endswith('.jpg')]
                batch_images = np.array(batch).astype(np.float32)

                start_time = time.time()
                vae_current_lr = learning_rate

                # update
                p, p1, p2, p3, kl, sse, errE, _ = sess.run([p_loss,p1_loss,p2_loss,p3_loss,KL_loss,SSE_loss,VAE_loss,vae_optim], feed_dict={input_imgs: batch_images, lr_vae:vae_current_lr})

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, vae_loss:%.8f, kl_loss:%.8f, sse_loss:%.8f, p1_loss:%.8f, p2_loss:%.8f, p3_loss:%.8f, p_loss:%.8f" \
                        % (epoch, epoch, idx, batch_idxs,
                            time.time() - start_time, errE, kl, sse, p1, p2, p3, p))
                sys.stdout.flush()

                iter_counter += 1
                # save samples
                if np.mod(iter_counter, sample_step) == 0:
                    # generate and visualize generated images
                    img1, img2 = sess.run([gen2.outputs, gen3.outputs], feed_dict={input_imgs: batch_images})

                    if save_sample_images:
                        recdir = os.path.join(get_current_record_dir(default_if_none=True), 'images')
                        print(f'Saving samples in Record Directory: {recdir}')
                        save_images(img1, [8, 8], '{}/train_{:02d}_{:04d}.png'.format(recdir, epoch, idx))
                        save_images(img2, [8, 8], '{}/train_{:02d}_{:04d}_random.png'.format(recdir, epoch, idx))
                        save_images(batch_images,[8, 8],'{}/input.png'.format(recdir))

                    with hold_dbplots():
                        dbplot(batch_images, 'batch_images', cornertext=f'Iteration {iter_counter}')
                        dbplot(img1, 'reconstructions')
                        dbplot(img2, 'random_samples')

                    print("[Sample] sample generated!!!")
                    sys.stdout.flush()

                # save checkpoint
                if np.mod(iter_counter, save_step) == 0:
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    net_e_name = os.path.join(save_dir, 'net_e.npz')
                    net_g_name = os.path.join(save_dir, 'net_g.npz')
                    # this version is for future re-check and visualization analysis
                    net_e_iter_name = os.path.join(save_dir, 'net_e_%d.npz' % iter_counter)
                    net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)

                    # params of two branches
                    net_out_params = net_out1.all_params + net_out2.all_params
                    # remove repeat params
                    net_out_params = tl.layers.list_remove_repeat(net_out_params)
                    tl.files.save_npz(net_out_params, name=net_e_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_name, sess=sess)

                    tl.files.save_npz(net_out_params, name=net_e_iter_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_iter_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")

                idx += 1
                # print idx
            except StopIteration:
                print('one epoch finished')
                break

    training_end_time = time.time()
    print("The processing time of program is : {:.2f}mins".format((training_end_time-training_start_time)/60.0))


if __name__ == '__main__':

    main()
