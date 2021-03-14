# coral-deeplab

[Coral Edge TPU](https://coral.ai/products/) compilable version of DeepLab v3 Plus

Implementation follows original paper as close as possible, while still being compatible with Edge TPU. Due to hardware limitations, max input shape has been decreases to 224x224, output stride is set at 16 and dilation rates in Atrous Spatial Pyramid Pooling branches are 3, 6, 9 instead of original 6, 12, 18. All 5 branches of ASPP are used. MobileNetV2 is used as encoder, but last 4 blocks had been modified to use atrous convolution in order to preserve spatial resolution.

## Instalation

```bash
git clone https://github.com/xadrianzetx/coral-deeplab.git
cd coral-deeplab
python setup.py install
```

## Usage

```python
import tensorflow as tf
import coral_deeplab as cdl

model = cdl.applications.CoralDeepLabV3Plus()
isinstance(model, tf.keras.Model)
# True
```

## References

* [Chen et al., 2018, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)