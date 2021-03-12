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

from coral_deeplab._blocks import (
    inverted_res_block,
    deeplab_aspp_module,
    deeplab_decoder
)


def CoralDeepLabV3Plus(input_shape: tuple, n_classes: int,
                       **kwargs) -> tf.keras.Model:
    """
    """

    encoder = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )

    encoder_maps = encoder.get_layer('block_3_expand_relu').output
    encoder_last = encoder.get_layer('block_13_expand_relu').output

    x = inverted_res_block(encoder_last, 160, expand_channels=576, block_num=13)
    x = inverted_res_block(x, 160, expand=True, skip=True, block_num=14)
    x = inverted_res_block(x, 160, expand=True, skip=True, block_num=15)
    aspp_in = inverted_res_block(x, 320, expand=True, block_num=16)

    dilation_rates = [3, 6, 9]
    bn_eps = 1e-5

    aspp_out = deeplab_aspp_module(aspp_in, dilation_rates, bn_eps)
    outputs = deeplab_decoder(aspp_out, encoder_maps, n_classes, bn_eps)
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs)

    return model
