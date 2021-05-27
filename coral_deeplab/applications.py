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

__all__ = ['CoralDeepLabV3']

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input


from coral_deeplab._encoders import mobilenetv2
from coral_deeplab._blocks import (
    deeplab_aspp_module,
    deeplabv3_decoder
)


def CoralDeepLabV3(input_shape: tuple = (513, 513, 3),
                   n_classes: int = 30, **kwargs) -> tf.keras.Model:
    """DeepLab v3 Plus implementation fully compilable to coral.ai Edge TPU.

    Implementation follows original paper as close as possible, while still
    being compatible with Edge TPU. Due to hardware limitations, max input
    shape has been decreases to 224x224x3, output stride is set at 16 and
    dilation rates in Atrous Spatial Pyramid Pooling branches are
    3, 6, 9 instead of original 6, 12, 18. All 5 branches of ASPP are used.

    MobileNetV2 is used as encoder, but last 4 blocks had been modified
    to use atrous convolution in order to preserve spatial resolution.

    Arguments
    ---------
    input_shape : tuple, default=(224, 224, 3)
        Input tensor shape.

    n_classes : int, default=30
        Number of segmentation classes.
        By default set to cityscapes dayaset
        number of class labels.

    Returns
    -------
    model : tf.keras.Model
        DeepLabV3Plus keras model instance.

    References
    ----------
    - [1] https://arxiv.org/pdf/1802.02611.pdf
    - [2] https://coral.ai/products/

    Notes
    -----
    There is no last activation layer. Model outputs logits.
    Default setup with input shape 224x224x3 and 30 output
    classes runs pretty close to compilable maximum. Setting
    grater input shape will result in error, and setting greater
    number of output classes might result in model not fully mapping
    to Edge TPU.

    Examples
    --------
    >>> import coral_deeplab as cdl
    >>> model = cdl.applications.CoralDeepLabV3Plus()
    >>> print(model.name)
    'CoralDeepLabV3Plus'
    """

    if np.argmin(input_shape) == 0:
        # assuming channels always
        # gonna be smallest number
        raise ValueError('Channels-first not supported.')

    if input_shape[0] != input_shape[1]:
        raise ValueError('Non square inputs not supported.')

    inputs = Input(shape=input_shape)
    aspp_in = mobilenetv2(inputs)
    aspp_out = deeplab_aspp_module(aspp_in)
    outputs = deeplabv3_decoder(aspp_out, input_shape[:2], n_classes)
    name = 'CoralDeeplabV3'
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model
