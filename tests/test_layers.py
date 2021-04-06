import unittest
import numpy as np
import tensorflow as tf

import coral_deeplab as cdl


class TestLayers(unittest.TestCase):

    def test_upsampling2d_out_shape(self):
        """Tests output shape after upsampling"""

        inputs = np.random.randn(1, 10, 10, 3)
        x = cdl.layers.UpSampling2DCompatV1((2, 2))(inputs)
        outputs = x.numpy()
        test = outputs.shape == (1, 20, 20, 3)
        self.assertTrue(test)

    def test_upsampling2d_invalid_mode(self):
        """Test behaviour when incorrect mode is passed"""

        with self.assertRaises(ValueError):
            inputs = np.random.randn(1, 10, 10, 3)
            cdl.layers.UpSampling2DCompatV1(interpolation='foo')(inputs)

    def test_upsampling2d_serialize_deserialize(self):
        """Test layer serialization and deserialization"""

        def testmodel():
            inputs = tf.keras.layers.Input((10, 10, 3))
            outputs = cdl.layers.UpSampling2DCompatV1((2, 2))(inputs)
            return tf.keras.Model(inputs=inputs, outputs=outputs)

        model = testmodel()
        model.compile(loss='binary_crossentropy', metrics=['acc'])

        try:
            model.save('model.h5')
            custom_objects = {'UpSampling2DCompatV1': cdl.layers.UpSampling2DCompatV1}
            tf.keras.models.load_model('model.h5', custom_objects=custom_objects)

        except NotImplementedError:
            self.fail('Model with UpSampling2DCompatV1 failed to serialize.')

        except TypeError:
            self.fail('Model with UpSampling2DCompatV1 failed to deserialize.')


if __name__ == 'main':
    unittest.main()
