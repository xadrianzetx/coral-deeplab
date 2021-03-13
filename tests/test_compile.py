import sys
import unittest
import numpy as np

import coral_deeplab as cdl


def fake_dataset_generator(shape, n_iter):
    def dataset():
        for _ in range(n_iter):
            data = np.random.randn(*shape)
            data *= (1 / 255)
            batch = np.expand_dims(data, axis=0)
            yield [batch.astype(np.float32)]
    return dataset


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

    @unittest.skipUnless(sys.platform.startswith('linux'), 'linux required')    
    def test_edgetpu_compiles(self):
        pass


if __name__ == '__main__':
    unittest.main()
