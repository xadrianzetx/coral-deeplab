import tensorflow as tf

from coral_deeplab.layers import UpSampling2D


def inverted_res_block():
    pass


def deeplab_aspp_module():
    pass


def deeplab_decoder(inputs: tf.Tensor, skip_con: tf.Tensor,
                    n_classes: int, bn_epsilon: float) -> tf.Tensor:
    """
    """

    if n_classes > 50:
        print('Warning - model might not compile'
              ' due to upsampling of large tensors.'
              ' Consider decreasing number of'
              ' segmentation classes.')

    skip = tf.keras.layers.Conv2D(48, 1, padding='same', use_bias=False,
                                  name='project_0')(skip_con)
    skip = tf.keras.layers.BatchNormalization(epsilon=bn_epsilon)(skip)
    skip = tf.keras.layers.ReLU()(skip)

    aspp_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(inputs)
    x = tf.keras.layers.Concatenate()([aspp_up, skip])

    x = tf.keras.layers.SeparableConv2D(256, 3, padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=bn_epsilon)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=bn_epsilon)(x)
    x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.SeparableConv2D(n_classes, 3, padding='same',
                                              use_bias=False)(x)
    outputs = UpSampling2D(size=(4, 4), interpolation='bilinear')(outputs)

    return outputs
