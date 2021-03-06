import os
import re
import sys
import uuid
import unittest
import subprocess
import numpy as np
import tensorflow as tf

import coral_deeplab as cdl


def fake_dataset_generator(shape, n_iter):
    def dataset():
        for _ in range(n_iter):
            data = np.random.randn(*shape)
            data *= (1 / 255)
            batch = np.expand_dims(data, axis=0)
            yield [batch.astype(np.float32)]
    return dataset


def quantize_and_compile(model, dataset):

    # setup tflite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # quantize
    quantized = converter.convert()
    model_name = f'{uuid.uuid4()}.tflite'
    model_path = os.path.join(os.getcwd(), model_name)
    open(model_path, 'wb').write(quantized)

    # compile
    cmd = ['edgetpu_compiler', '-s', model_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    stdout = p.stdout.read()

    return stdout.decode()


class TestCoralDeepLabV3Plus(unittest.TestCase):

    def test_channel_first(self):
        """Test if exception is rised when channel-first used"""

        with self.assertRaises(ValueError):
            input_shape = (3, 224, 224)
            cdl.applications.CoralDeepLabV3Plus(input_shape)

    def test_input_not_square(self):
        """Test if exception is rised when input not square"""

        with self.assertRaises(ValueError):
            input_shape = (112, 224, 3)
            cdl.applications.CoralDeepLabV3Plus(input_shape)

    def test_input_too_big(self):
        """Test if exception when input size is over max"""

        with self.assertRaises(ValueError):
            input_shape = (512, 512, 3)
            cdl.applications.CoralDeepLabV3Plus(input_shape)

    def test_input_match_output(self):
        """Test if output shape matches input"""

        supported_shapes = [96, 128, 160, 192, 224]
        for shape in supported_shapes:
            input_shape = (shape, shape, 3)
            model = cdl.applications.CoralDeepLabV3Plus(input_shape)
            _, *out_shape, _ = model.output_shape
            self.assertEqual(tuple(out_shape), input_shape[:-1])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'linux required')
    def test_edgetpu_compiles(self):
        """Test if model compiles to Edge TPU across input ranges"""

        supported_shapes = [96, 128, 160, 192, 224]
        for shape in supported_shapes:
            input_shape = (shape, shape, 3)
            model = cdl.applications.CoralDeepLabV3Plus(input_shape)
            datagen = fake_dataset_generator(input_shape, 10)
            stdout = quantize_and_compile(model, datagen)
            compiled = re.findall('Model compiled successfully', stdout)

            if not compiled:
                msg = f'Model not compiled for shape {input_shape}'
                self.fail(msg)


if __name__ == '__main__':
    unittest.main()
