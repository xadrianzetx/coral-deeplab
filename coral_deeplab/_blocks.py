# MIT License

# Copyright (c) 2021 xadrianzetx

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Concatenate,
    Add,
    Lambda,
    ReLU
)

from coral_deeplab.layers import UpSampling2DCompatV1


L2 = 4e-5
BN_MOMENTUM = 0.9997


def inverted_res_block(inputs: tf.Tensor, project_channels: int,
                       expand_channels: int, block_num: int,
                       strides: int = 1, dilation: int = 1,
                       skip: bool = False) -> tf.Tensor:
    """Modified MobileNetV2 inverted residual block.

    This implementation uses dilated convolution in its depthwise
    layer, which preserves spatial dimentions throught the layers
    (image has the same size in and out).

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    project_channels : int
        Number of feature maps to produce
        in project layer.

    block_num : int
        Residual block number
        (used for layer naming).

    expand_channels : int, default=960
        Number of feature maps to produce
        in expand layer (if used).

    expand : bool, default=False
        If true, expand layer is used in block.

    skip : bool, default=False
        If true, skip connection is used in block.

    Returns
    -------
    x : tf.Tensor
        Output tensor.
    """

    # expand
    x = Conv2D(expand_channels, 1, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(L2),
               name=f'{block_num}_expand')(inputs)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'{block_num}_expand_bn')(x)
    x = ReLU(6, name=f'{block_num}_expand_relu')(x)

    # depthwise
    x = DepthwiseConv2D(3, strides=strides, padding='same',
                        dilation_rate=dilation, use_bias=False,
                        depthwise_regularizer=tf.keras.regularizers.l2(L2),
                        name=f'{block_num}_depthwise')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'{block_num}_depthwise_bn')(x)
    x = ReLU(6, name=f'{block_num}_depthwise_relu')(x)

    # project
    x = Conv2D(project_channels, 1, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(L2),
               name=f'{block_num}_project')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'{block_num}_project_bn')(x)

    if skip:
        x = Add(name=f'{block_num}_add')([x, inputs])

    return x


def deeplab_aspp_module(inputs: tf.Tensor, bn_epsilon: float) -> tf.Tensor:
    """Implements Atrous Spatial Pyramid Pooling module.

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    bn_epsilon : float
        Epsilon used in batch normalization layer.

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.
    """

    # aspp branch 0
    b0 = SeparableConv2D(256, 3, padding='same',
                         use_bias=False, name='aspp0')(inputs)
    b0 = BatchNormalization(epsilon=bn_epsilon, name='aspp0_bn')(b0)
    b0 = ReLU(name='aspp0_relu')(b0)

    # branch 4
    _, *size, _ = tf.keras.backend.int_shape(inputs)
    b4 = GlobalAveragePooling2D(name='aspp4_pooling')(inputs)
    b4 = Lambda(lambda t: t[:, tf.newaxis, tf.newaxis, :])(b4)
    b4 = UpSampling2DCompatV1(output_shape=size, interpolation='bilinear')(b4)
    b4 = Conv2D(256, 1, padding='same', use_bias=False, name='aspp4')(b4)
    b4 = BatchNormalization(epsilon=bn_epsilon, name='aspp4_bn')(b4)
    b4 = ReLU(name='aspp4_relu')(b4)

    # concat and pointwise conv
    x = Concatenate(name='aspp_concat')([b0, b4])
    x = Conv2D(256, 1, padding='same', use_bias=False, name='aspp')(x)
    x = BatchNormalization(epsilon=bn_epsilon, name='aspp_bn')(x)
    outputs = ReLU(name='aspp_relu')(x)

    return outputs


def deeplab_decoder(inputs: tf.Tensor, skip_con: tf.Tensor,
                    output_shape: tuple, n_classes: int,
                    bn_epsilon: float) -> tf.Tensor:
    """Implements DeepLabV3Plus decoder module.

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    skip_con : tf.Tensor
        Encoder tensor used
        in skip connection.

    n_classes : int
        Final number of feature maps.

    bn_epsilon : float
        Epsilon used in batch
        normalization layer.

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.

    Notes
    -----
    This implementation is using output stride 16.
    Outputs logits. There is no final activation layer.
    """

    if n_classes > 50:
        print('Warning - model might not compile'
              ' due to upsampling of large tensors.'
              ' Consider decreasing number of'
              ' segmentation classes.')

    skip = Conv2D(48, 1, padding='same', use_bias=False,
                  name='project_0')(skip_con)
    skip = BatchNormalization(epsilon=bn_epsilon)(skip)
    skip = ReLU()(skip)

    _, *skip_shape, _ = tf.keras.backend.int_shape(skip)
    aspp_up = UpSampling2DCompatV1(output_shape=skip_shape,
                                   interpolation='bilinear')(inputs)
    x = Concatenate()([aspp_up, skip])

    x = SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=bn_epsilon)(x)
    x = ReLU()(x)

    x = SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=bn_epsilon)(x)
    x = ReLU()(x)

    outputs = SeparableConv2D(n_classes, 3, padding='same', use_bias=False)(x)
    outputs = UpSampling2DCompatV1(output_shape=output_shape,
                                   interpolation='bilinear')(outputs)

    return outputs
