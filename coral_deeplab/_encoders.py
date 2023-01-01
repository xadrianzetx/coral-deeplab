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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ReLU

from coral_deeplab._blocks import BN_MOMENTUM
from coral_deeplab._blocks import L2
from coral_deeplab._blocks import inverted_res_block


def mobilenetv2(inputs: tf.Tensor, alpha: float = 1.0) -> tf.Tensor:
    """Implements modified version of MobileNetV2.

    Last three inverted res block use dilated conv
    in depthwise convolution layers to preserve spatial
    resolution of the feature maps.

    Arguments
    ---------
    inputs : tf.Tensor
        Input tensor.

    alpha : float, default=1.0
        Depth multiplier.

    Returns
    -------
    outputs : tf.Tensor
        Output tensor.
    """

    x = Conv2D(
        int(alpha * 32),
        3,
        2,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(L2),
        name="Conv",
    )(inputs)
    x = BatchNormalization(momentum=BN_MOMENTUM, name="Conv/BatchNorm")(x)
    x = ReLU(6, name="Conv/relu")(x)

    x = DepthwiseConv2D(
        3,
        strides=1,
        padding="same",
        dilation_rate=1,
        depthwise_regularizer=tf.keras.regularizers.L2(L2),
        use_bias=False,
        name="expanded_conv/depthwise",
    )(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, name="expanded_conv/depthwise/BatchNorm")(x)
    x = ReLU(6, name="expanded_conv/depthwise/relu")(x)

    x = Conv2D(
        int(alpha * 16),
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.L2(L2),
        name="expanded_conv/project",
    )(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, name="expanded_conv/project/BatchNorm")(x)

    x = inverted_res_block(
        x, int(alpha * 24), expand_channels=int(alpha * 96), strides=2, block_num=1
    )
    x = inverted_res_block(
        x, int(alpha * 24), expand_channels=int(alpha * 144), skip=True, block_num=2
    )
    x = inverted_res_block(
        x, int(alpha * 32), expand_channels=int(alpha * 144), strides=2, block_num=3
    )
    x = inverted_res_block(
        x, int(alpha * 32), expand_channels=int(alpha * 192), skip=True, block_num=4
    )
    x = inverted_res_block(
        x, int(alpha * 32), expand_channels=int(alpha * 192), skip=True, block_num=5
    )
    x = inverted_res_block(
        x, int(alpha * 64), expand_channels=int(alpha * 192), strides=2, block_num=6
    )
    x = inverted_res_block(
        x, int(alpha * 64), expand_channels=int(alpha * 384), skip=True, block_num=7
    )
    x = inverted_res_block(
        x, int(alpha * 64), expand_channels=int(alpha * 384), skip=True, block_num=8
    )
    x = inverted_res_block(
        x, int(alpha * 64), expand_channels=int(alpha * 384), skip=True, block_num=9
    )
    x = inverted_res_block(x, int(alpha * 96), expand_channels=int(alpha * 384), block_num=10)
    x = inverted_res_block(
        x, int(alpha * 96), expand_channels=int(alpha * 576), skip=True, block_num=11
    )
    x = inverted_res_block(
        x, int(alpha * 96), expand_channels=int(alpha * 576), skip=True, block_num=12
    )
    x = inverted_res_block(x, int(alpha * 160), expand_channels=int(alpha * 576), block_num=13)

    # modified
    x = inverted_res_block(
        x, int(alpha * 160), expand_channels=int(alpha * 960), skip=True, dilation=2, block_num=14
    )
    x = inverted_res_block(
        x, int(alpha * 160), expand_channels=int(alpha * 960), skip=True, dilation=2, block_num=15
    )
    outputs = inverted_res_block(
        x, int(alpha * 320), expand_channels=int(alpha * 960), dilation=2, block_num=16
    )

    return outputs
