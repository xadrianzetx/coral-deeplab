# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of semantic segmentation.
The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:
```
bash examples/install_requirements.sh semantic_segmentation.py
python3 examples/semantic_segmentation.py \
  --model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
  --input test_data/bird.bmp \
  --keep_aspect_ratio \
  --output ${HOME}/segmentation_result.jpg
```
"""

import argparse

import numpy as np
from PIL import Image

import cv2
import coral_deeplab as cdl
import tflite_runtime.interpreter as tflite


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
      label: A 2D array with integer type, storing the segmentation label.
    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def output_tensor(interpreter, i):
    """Gets a model's ith output tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      i (int): The index position of an output tensor.
    Returns:
      The output tensor at the specified position.
    """
    return interpreter.tensor(interpreter.get_output_details()[i]["index"])()


def input_details(interpreter, key):
    """Gets a model's input details by specified key.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
      key (int): The index position of an input tensor.
    Returns:
      The input details.
    """
    return interpreter.get_input_details()[0][key]


def input_size(interpreter):
    """Gets a model's input size as (width, height) tuple.
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor size as (width, height) tuple.
    """
    _, height, width, _ = input_details(interpreter, "shape")
    return width, height


def input_tensor(interpreter):
    """Gets a model's input tensor view as numpy array of shape (height, width, 3).
    Args:
      interpreter: The ``tf.lite.Interpreter`` holding the model.
    Returns:
      The input tensor view as :obj:`numpy.array` (height, width, 3).
    """
    tensor_index = input_details(interpreter, "index")
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, data):
    """Copies data to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      data: The input tensor.
    """
    input_tensor(interpreter)[:, :] = data


def set_resized_input(interpreter, size, resize):
    """Copies a resized and properly zero-padded image to a model's input tensor.
    Args:
      interpreter: The ``tf.lite.Interpreter`` to update.
      size (tuple): The original image size as (width, height) tuple.
      resize: A function that takes a (width, height) tuple, and returns an
        image resized to those dimensions.
    Returns:
      The resized tensor with zero-padding as tuple
      (resized_tensor, resize_ratio).
    """
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = np.reshape(result, (h, w, channel))
    return result, (scale, scale)


def get_output(interpreter):
    output_details = interpreter.get_output_details()[0]
    return interpreter.tensor(output_details["index"])()[0].astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    #   parser.add_argument('--model', required=True,
    #                       help='Path of the segmentation model.')
    parser.add_argument("--input", required=True, help="File path of the input image.")
    parser.add_argument(
        "--output",
        default="semantic_segmentation_result.jpg",
        help="File path of the output image.",
    )
    parser.add_argument(
        "--keep_aspect_ratio",
        action="store_true",
        default=False,
        help=(
            "keep the image aspect ratio when down-sampling the image by adding "
            "black pixel padding (zeros) on bottom or right. "
            "By default the image is resized and reshaped without cropping. This "
            "option should be the same as what is applied on input images during "
            "model training. Otherwise the accuracy may be affected and the "
            "bounding box of detection result may be stretched."
        ),
    )
    args = parser.parse_args()

    model = cdl.from_precompiled(cdl.pretrained.EdgeTPUModel.DEEPLAB_V3_DM1)
    interpreter = tflite.Interpreter(
        model, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    width, height = input_size(interpreter)

    img = Image.open(args.input)
    if args.keep_aspect_ratio:
        resized_img, _ = set_resized_input(
            interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS)
        )
    else:
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        set_input(interpreter, resized_img)

    interpreter.invoke()

    result = get_output(interpreter)
    result = cv2.resize(result, (height, width))
    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

    # # If keep_aspect_ratio, we need to remove the padding area.
    new_width, new_height = resized_img.size
    result = result[:new_height, :new_width]
    mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

    # Concat resized input image and processed segmentation results.
    output_img = Image.new("RGB", (2 * new_width, new_height))
    output_img.paste(resized_img, (0, 0))
    output_img.paste(mask_img, (width, 0))
    output_img.save(args.output)
    print("Done. Results saved at", args.output)


if __name__ == "__main__":
    main()
