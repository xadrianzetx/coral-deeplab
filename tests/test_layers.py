import unittest
import numpy as np

import coral_deeplab as cdl


class TestLayers(unittest.TestCase):

    def test_upsampling2d_out_shape(self):
        """Tests output shape after upsampling"""

        inputs = np.random.randn(1, 10, 10, 3)
        x = cdl.layers.UpSampling2D((2, 2))(inputs)
        outputs = x.numpy()
        test = outputs.shape == (1, 20, 20, 3)
        self.assertTrue(test)

    def test_upsampling2d_invalid_mode(self):
        """Test behaviour when incorrect mode is passed"""

        with self.assertRaises(ValueError):
            inputs = np.random.randn(1, 10, 10, 3)
            cdl.layers.UpSampling2D(interpolation='foo')(inputs)


if __name__ == 'main':
    unittest.main()
