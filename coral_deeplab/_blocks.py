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
    DepthwiseConv2D,
    BatchNormalization,
    AveragePooling2D,
    Concatenate,
    Add,
    Lambda,
    ReLU
)


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

    expand_channels : int
        Number of feature maps to produce
        in expand layer.

    block_num : int
        Residual block number
        (used for layer naming).

    strides : int
        Depthwise convolution striding.

    dilation : int
        Depthwise convolution dilation.

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
               name=f'expanded_conv_{block_num}/expand')(inputs)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'expanded_conv_{block_num}/expand/BatchNorm')(x)
    x = ReLU(6, name=f'expanded_conv_{block_num}/expand/relu')(x)

    # depthwise
    x = DepthwiseConv2D(3, strides=strides, padding='same',
                        dilation_rate=dilation, use_bias=False,
                        depthwise_regularizer=tf.keras.regularizers.l2(L2),
                        name=f'expanded_conv_{block_num}/depthwise')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'expanded_conv_{block_num}/depthwise/BatchNorm')(x)
    x = ReLU(6, name=f'expanded_conv_{block_num}/depthwise/relu')(x)

    # project
    x = Conv2D(project_channels, 1, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(L2),
               name=f'expanded_conv_{block_num}/project')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name=f'expanded_conv_{block_num}/project/BatchNorm')(x)

    if skip:
        x = Add(name=f'{block_num}_add')([inputs, x])

    return x


def deeplab_aspp_module(inputs: tf.Tensor) -> tf.Tensor:
    """Implements Atrous Spatial Pyramid Pooling module.

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.
    """

    # aspp branch 0
    b0 = Conv2D(256, 1, padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(L2), name='aspp0')(inputs)
    b0 = BatchNormalization(momentum=BN_MOMENTUM, name='aspp0/BatchNorm')(b0)
    b0 = ReLU(name='aspp0/relu')(b0)

    # branch 4
    _, *size, _ = tf.keras.backend.int_shape(inputs)
    b4 = AveragePooling2D(pool_size=size, strides=(1, 1))(inputs)
    b4 = Conv2D(256, 1, padding='same', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(L2),
                name='image_pooling')(b4)
    b4 = BatchNormalization(momentum=BN_MOMENTUM,
                            name='image_pooling/BatchNorm')(b4)
    b4 = ReLU(name='image_pooling/relu')(b4)
    b4 = Lambda(lambda t: tf.compat.v1.image.resize_bilinear(t, size=size, align_corners=True))(b4)

    # concat and pointwise conv
    x = Concatenate(name='aspp_concat')([b4, b0])
    x = Conv2D(256, 1, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.L2(L2),
               name='concat_projection')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name='concat_projection/BatchNorm')(x)
    outputs = ReLU(name='concat_projection/relu')(x)

    return outputs


def deeplabv3_decoder(inputs: tf.Tensor, n_classes: int) -> tf.Tensor:
    """Implements DeepLabV3 decoder module.

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    n_classes : int
        Number of segmentation classes to output
        from the decoder.

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.
    """

    outputs = Conv2D(n_classes, 1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.L2(L2),
                     name='logits/semantic')(inputs)

    return outputs


def deeplabv3plus_decoder(inputs: tf.Tensor, skip_con: tf.Tensor,
                          n_classes: int) -> tf.Tensor:
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

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.
    """

    skip = Conv2D(48, 1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.L2(L2),
                  use_bias=False, name='skip_con_conv')(skip_con)
    skip = BatchNormalization(momentum=BN_MOMENTUM,
                              name='skip_con_conv/BatchNorm')(skip)
    skip = ReLU(name='skip_con_conv/relu')(skip)

    _, *size, _ = tf.keras.backend.int_shape(skip)
    aspp = Lambda(
        lambda t: tf.compat.v1.image.resize_bilinear(
            t, size=size, align_corners=True), name='aspp_resize')(inputs)

    x = tf.keras.layers.Concatenate(name='decoder_concat')([aspp, skip])

    x = Conv2D(256, 3, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.L2(L2),
               name='decoder_conv_1')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name='decoder_conv_1/BatchNorm')(x)
    x = ReLU(name='decoder_conv_1/relu')(x)

    x = Conv2D(256, 3, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.L2(L2),
               name='decoder_conv_2')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM,
                           name='decoder_conv_2/BatchNorm')(x)
    x = ReLU(name='decoder_conv_2/relu')(x)

    outputs = Conv2D(n_classes, 1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.L2(L2),
                     name='logits/semantic')(x)

    return outputs
