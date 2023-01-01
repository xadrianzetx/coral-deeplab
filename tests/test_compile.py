import os
import re
import subprocess
import sys
import unittest
import uuid

import numpy as np
import tensorflow as tf

import coral_deeplab as cdl


def fake_dataset_generator(shape, n_iter):
    def dataset():
        for _ in range(n_iter):
            data = np.random.randn(*shape)
            data *= 1 / 255
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
    model_name = f"{uuid.uuid4()}.tflite"
    model_path = os.path.join(os.getcwd(), model_name)
    open(model_path, "wb").write(quantized)

    # compile
    cmd = ["edgetpu_compiler", "-a", "-s", model_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    stdout = p.stdout.read()

    return stdout.decode()


class TestCoralDeepLabV3(unittest.TestCase):
    def test_channel_first(self):
        """Test if exception is rised when channel-first used"""

        with self.assertRaises(ValueError):
            input_shape = (3, 224, 224)
            cdl.applications.CoralDeepLabV3(input_shape)

    def test_input_not_square(self):
        """Test if exception is rised when input not square"""

        with self.assertRaises(ValueError):
            input_shape = (112, 224, 3)
            cdl.applications.CoralDeepLabV3(input_shape)

    def test_dm1_voc_weights_load(self):
        """Test if pre-trained dm1 voc weights load."""

        try:
            cdl.applications.CoralDeepLabV3(weights="pascal_voc")

        except Exception as e:
            self.fail(str(e))

    @unittest.skipUnless(sys.platform.startswith("linux"), "linux required")
    def test_dlv3_edgetpu_compiles(self):
        """Test if model compiles to Edge TPU across input ranges"""

        supported_shapes = [192, 224, 513]
        for shape in supported_shapes:
            input_shape = (shape, shape, 3)
            model = cdl.applications.CoralDeepLabV3(input_shape)
            datagen = fake_dataset_generator(input_shape, 10)
            stdout = quantize_and_compile(model, datagen)
            compiled = re.findall("Model compiled successfully", stdout)

            if not compiled:
                msg = f"Model not compiled for shape {input_shape}"
                self.fail(msg)

    @unittest.skip("Finetuning was temporarily disabled.")
    def test_dlv3_pretrained_edgetpu_compile(self):
        """Test if model compiles from pretrained weights"""

        alphas = [0.5, 1.0]
        for alpha in alphas:
            model = cdl.applications.CoralDeepLabV3(weights="pascal_voc", alpha=alpha)
            datagen = fake_dataset_generator((513, 513, 3), 10)
            stdout = quantize_and_compile(model, datagen)
            compiled = re.findall("Model compiled successfully", stdout)

            if not compiled:
                self.fail("Pretrained model not compiled")

    @unittest.skipUnless(sys.platform.startswith("linux"), "linux required")
    def test_dlv3plus_edgetpu_compile(self):
        """Test if DeepLabV3Plus model compiles to Edge TPU across input ranges"""

        supported_shapes = [192, 224, 513]
        for shape in supported_shapes:
            input_shape = (shape, shape, 3)
            model = cdl.applications.CoralDeepLabV3Plus(input_shape)
            datagen = fake_dataset_generator(input_shape, 10)
            stdout = quantize_and_compile(model, datagen)
            compiled = re.findall("Model compiled successfully", stdout)

            if not compiled:
                self.fail(f"Failed to compile DeepLabV3Plus with input shape {input_shape}")

    @unittest.skip("Finetuning was temporarily disabled.")
    def test_dlv3plus_pretrained_edgetpu_compile(self):
        """Test if pretrained DeepLabV3Plus model compiles to Edge TPU"""

        alphas = [0.5, 1.0]
        for alpha in alphas:
            model = cdl.applications.CoralDeepLabV3Plus(alpha=alpha, weights="pascal_voc")
            datagen = fake_dataset_generator((513, 513, 3), 10)
            stdout = quantize_and_compile(model, datagen)
            compiled = re.findall("Model compiled successfully", stdout)

            if not compiled:
                self.fail(f"Failed to compile pretrainrd DeepLabV3Plus with alpha {alpha}")


if __name__ == "__main__":
    unittest.main()
