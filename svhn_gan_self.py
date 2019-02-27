import os, sys
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
import nn
init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)

class GAN_manifold_reg():
    def __init__(self):
        pass

    def leakyReLu(self, x, alpha=0.2, name=None):
        if name:
            with tf.variable_scope(name):
                return self._leakyReLu_impl(x, alpha)
        else:
            return self._leakyReLu_impl(x, alpha)

    def _leakyReLu_impl(self, x, alpha):
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def classifier(self, inp, is_training, init=False, reuse=False, getter=None):
        with tf.variable_scope('classifier', reuse=reuse, custom_getter=getter):
            counter = {}
            x = tf.reshape(inp, [-1, 32, 32, 3])

            x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

            x = nn.conv2d(x, 64, nonlinearity = self.leakyReLu, init=init, counters=counter)
            x = nn.conv2d(x, 64, nonlinearity= self.leakyReLu, init=init, counters=counter)
            x = nn.conv2d(x, 64, stride=[2, 2], nonlinearity= self.leakyReLu, init=init, counters=counter)

            x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

            x = nn.conv2d(x, 128, nonlinearity = self.leakyReLu, init=init, counters=counter)
            x = nn.conv2d(x, 128, nonlinearity = self.leakyReLu, init=init, counters=counter)
            x = nn.conv2d(x, 128, stride=[2, 2], nonlinearity = self.leakyReLu, init=init, counters=counter)

            x = tf.layers.dropout(x, rate=0.5, training = is_training, name='dropout_2')

            x = nn.conv2d(x, 128, pad='VALID', nonlinearity = self.leakyReLu, init=init, counters=counter)
            x = nn.nin(x, 128, counters=counter, nonlinearity = self.leakyReLu, init=init)
            x = nn.nin(x, 128, counters=counter, nonlinearity = self.leakyReLu, init=init)
            x = tf.layers.max_pooling2d(x, pool_size = 6, strides=1,
                                        name='avg_pool_0')
            x = tf.squeeze(x, [1, 2])

            intermediate_layer = x

            logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter, init_scale=0.1)

            return logits, intermediate_layer

    def bad_generator(self, z_seed, is_training, init=False, reuse=False):
        with tf.variable_scope('bad_generator', reuse=reuse):
            counter = {}
            x = z_seed
            with tf.variable_scope('dense_1'):
                # x = tf.layers.dense(x, units=4 * 4 * 512, kernel_initializer = init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
                x = nn._linear_fc(x, 4 * 4 * 512, 'bg_h0_lin')
                x = nn.batch_norm_contrib(x, name='batchnorm_1', train = is_training)
                x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, 4, 4, 512])

            with tf.variable_scope('deconv_1'):
                # x = tf.layers.conv2d_transpose(x, 256, [5, 5], strides=[2, 2], padding='SAME',
                #                                kernel_initializer=init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
                x = nn._deconv2d(x, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='bg_dconv0')  # [8, 8]
                x = nn.batch_norm_contrib(x, name='batchnorm_2', train = is_training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv_2'):
                # x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=[2, 2], padding='SAME',
                #                                kernel_initializer=init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
                x = nn._deconv2d(x, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='bg_dconv1')
                x = nn.batch_norm_contrib(x, name='batchnorm_2', train = is_training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv_3'):
                output = nn.deconv2d(x, num_filters=3, filter_size=[5, 5], stride=[2, 2], nonlinearity=tf.tanh,
                                     init=init,
                                     counters=counter, init_scale=0.1)
        return output

    def bad_sampler(self, z_seed, is_training = False, init = False):
        with tf.variable_scope('bad_generator', reuse=tf.AUTO_REUSE):
            counter = {}
            x = z_seed
            with tf.variable_scope('dense_1'):
                # x = tf.layers.dense(x, units=4 * 4 * 512, kernel_initializer = init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
                x = nn._linear_fc(x, 4 * 4 * 512, 'bg_h0_lin')
                x = nn.batch_norm_contrib(x, name='batchnorm_1', train = is_training)
                x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, 4, 4, 512])

            with tf.variable_scope('deconv_1'):
                # x = tf.layers.conv2d_transpose(x, 256, [5, 5], strides=[2, 2], padding='SAME',
                #                                kernel_initializer=init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
                x = nn._deconv2d(x, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='bg_dconv0')  # [8, 8]
                x = nn.batch_norm_contrib(x, name='batchnorm_2', train = is_training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv_2'):
                # x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=[2, 2], padding='SAME',
                #                                kernel_initializer=init_kernel)
                # x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
                x = nn._deconv2d(x, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='bg_dconv1')
                x = nn.batch_norm_contrib(x, name='batchnorm_2', train = is_training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv_3'):
                output = nn.deconv2d(x, num_filters=3, filter_size=[5, 5], stride=[2, 2], nonlinearity=tf.tanh,
                                     init=init,
                                     counters=counter, init_scale=0.1)
        return output

    def forward_pass(self, z_b, z_b_pert, x_l_c, x_u_c, train):
        """
        :param z: latent variable [200, 100]
        :param x_l_c: [200, 32, 32, 3]
        :param y_l_c: [200, 10]
        :param x_l_d: [20, 32, 32, 3]
        :param y_l_d: [20, 10]
        :param x_u_d: [180, 32, 32, 3]
        :param x_u_c: [200, 32, 32, 3]
        :return:
        """
        # output of G
        self.bad_generator(z_b, train, init=True)  # init of weightnorm weights cf Salimans et al.
        G = self.bad_generator(z_b, train, init = False, reuse = True)
        G_pert = self.bad_generator(z_b_pert, train, init = False, reuse=True)

        # output of C for real images
        self.classifier(x_u_c, train, init = True) # init of weightnorm weights cf Salimans et al.
        C_real_logits, feat_real = self.classifier(x_l_c, train, init = False, reuse = True)

        # output of C for unlabel images (as false examples to D)
        C_unl_logits, feat_unl = self.classifier(x_u_c, train, init = False, reuse = True)

        # output of G for generated images
        C_fake_logits, feat_fake = self.classifier(G, train, init = False, reuse = True)
        C_fake_pert_logits, feat_fake_pert = self.classifier(G_pert, train, init = False, reuse = True)

        return [[G, G_pert], [C_real_logits, C_unl_logits, C_fake_logits, C_fake_pert_logits, feat_real, feat_unl, \
                              feat_fake, feat_fake_pert]]

if __name__ == "__main__":
    pass