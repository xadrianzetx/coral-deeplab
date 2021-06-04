# coral-deeplab

[Coral Edge TPU](https://coral.ai/products/) compilable version of DeepLab v3 implemented in `tf.keras` with pretrained weights and Edge TPU pre-compiled models included.

Implementation follows original paper as close as possible, while still being compatible with Edge TPU. The only difference is that last upsampling layer has been removed from decoder due to performance reasons. Thanks to multi subgraph support in `edgetpu_compiler`, model runs almost all operations on TPU, where [original model](https://coral.ai/models/semantic-segmentation/) delegates entirety of decoder to run on CPU.

Thanks to pretrained weights and `tf.keras` implementation it's easy to fine tune this model, or even train it from scratch.

## Instalation

```bash
git clone https://github.com/xadrianzetx/coral-deeplab.git
cd coral-deeplab
python setup.py install
```

## Usage

You can train from scratch...

```python
import tensorflow as tf
import coral_deeplab as cdl

model = cdl.applications.CoralDeepLabV3()
isinstance(model, tf.keras.Model)
# True
```

...finetune...

```python
import tensorflow as tf
import coral_deeplab as cdl

model = cdl.applications.CoralDeepLabV3(weights='pascal_voc')
isinstance(model, tf.keras.Model)
# True
```

...or just pull pre-compiled Edge TPU models straight to interpreter.

```python
# TBA :^)
```

## References

* [Chen et al., 2017, Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* [Chen et al., 2018, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)