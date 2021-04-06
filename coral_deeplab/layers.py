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

import numpy as np
import tensorflow as tf


class UpSampling2DCompatV1(tf.keras.layers.Layer):
    """Upsampling layer using tensorflow
    v1 resize implementation.

    Arguments
    ---------
    size : tuple, default=(2, 2)
        Upsampling factors for rows and columns.

    interpolation : str, default='nearest'
        Resize method. One of: nearest, bilinear, bicubic.

    align_corners: bool, default=False
        If True, the centers of the 4 corner
        pixels of the input and output tensors
        are aligned, preserving the values
        at the corner pixels.
    """

    def __init__(self, size: tuple = (2, 2),
                 interpolation: str = 'nearest',
                 align_corners: bool = False,
                 **kwargs):

        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.align_corners = align_corners

    def build(self, input_shape: tuple):

        interpolations = ['nearest', 'bilinear', 'cubic']
        if self.interpolation not in interpolations:
            raise ValueError('Interpolation should be one of'
                             f'{interpolations}, got {self.interpolation}')

        oldsize = input_shape[1:3]
        self.newsize = [np.prod(s) for s in zip(self.size, oldsize)]

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:

        return tf.compat.v1.image.resize(
            inputs,
            self.newsize,
            method=self.interpolation,
            align_corners=self.align_corners
        )

    def get_config(self) -> dict:

        config = super().get_config()
        config.update({
            'size': self.size,
            'interpolation': self.interpolation,
            'align_corners': self.align_corners
        })

        return config
