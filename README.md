# coral-deeplab

[Coral Edge TPU](https://coral.ai/products/) compilable version of DeepLab v3 and DeepLab v3 Plus implemented in `tf.keras` with pretrained weights and Edge TPU pre-compiled models included.

Implementation follows original paper as close as possible, while still being compatible with Edge TPU. The only difference is that last upsampling layer has been removed from decoder due to performance reasons. Thanks to multi subgraph support in `edgetpu_compiler`, model runs almost all operations on TPU, where [original model](https://coral.ai/models/semantic-segmentation/) delegates entirety of decoder to run on CPU.

Thanks to pretrained weights and `tf.keras` implementation it's easy to fine tune this model, or even train it from scratch.

## Instalation

```bash
pip install git+https://github.com/xadrianzetx/coral-deeplab.git
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

...or just pull pre-compiled Edge TPU models straight to interpreter (can use `--no-deps` for lightweight install).

```python
import coral_deeplab as cdl
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(
    cdl.from_precompiled(cdl.pretrained.EdgeTPUModel.DEEPLAB_V3_DM1),
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1)]
)
```

## Latency

All models have been tested with 513x513 input on fresh install of `Raspbian GNU/Linux 10 (buster)`.

|Model|Backbone|Depth mul.|ARM Cortex-A53 (RPi3B+)|ARM Cortex-A72 (RPi4)|
|---|:---:|:---:|:---:|:---:|
|[DeepLab v3 (google)](https://coral.ai/models/semantic-segmentation/)|MobileNet v2|1.0|~368.6 ms/iter (~2.7 FPS)|~147.5 ms/iter (~6.8 FPS)|
|DeepLab v3 (this)|MobileNet v2|1.0|~143.0 ms/iter (~7.0 FPS)|~43.7 ms/iter (~22.9 FPS)|
|DeepLab v3 (this)|MobileNet v2|0.5|~129.4 ms/iter (~7.8 FPS)|~32.6 ms/iter (~30.6 FPS)|

## Results

From left: original image, DeepLab v3 (google), DeepLab v3 (this). All segmentation masks were produced with Edge TPU.

<p align="center">
<image src="https://drive.google.com/uc?id=1fRJWmODwjHUgdKwCPvM8H4ZEpvzGuxlQ"></image>
</p>

## References

* [Chen et al., 2017, Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* [Chen et al., 2018, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)