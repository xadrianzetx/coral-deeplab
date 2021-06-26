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

from enum import Enum


class MLModel(Enum):
    pass


class KerasModel(MLModel):
    DEEPLAB_V3_DM1 = {
        'origin': '1CE7cMfgViNgFxXKbCq0wFXeO8slV0Z01',
        'filename': 'deeplabv3_mnv2_dm1_voc_tainaug_os16.h5',
        'checksum': 'b326724d7e89d8cc7f409edbf1b11105'
    }
    DEEPLAB_V3_DM05 = {
        'origin': '1J-8hCUNYxbWgazflv8CGVYgmqoFHxB_N',
        'filename': 'deeplabv3_mnv2_dm05_voc_tainaug_os16.h5',
        'checksum': '36e1e957a62848451db92c53abc1d7d7'
    }
    DEEPLAB_V3_PLUS_DM1 = {
        'origin': '191I3qg-S245BD8aX1jfGF2Yy3H9-1A1l',
        'filename': 'deeplabv3plus_mnv2_dm1_voc_trainaug_os4.h5',
        'checksum': 'c43f0acf3a256daa237da66ecedb4565'
    }
    DEEPLAB_V3_PLUS_DM05 = {
        'origin': '17wv_wRPZMnj2s_y_nol8whkwda0C2O77',
        'filename': 'deeplabv3plus_mnv2_dm05_voc_trainaug_os4.h5',
        'checksum': 'e3e002c39716bc54f966bae657fc2f78'
    }


class EdgeTPUModel(MLModel):
    DEEPLAB_V3_DM1 = {
        'origin': '1YmaaQ9qOxlMfB9eAI7roOqgeo4y7Mosg',
        'filename': 'deeplabv3_mnv2_dm1_voc_tainaug_os16_edgetpu.tflite',
        'checksum': '6c0ade5b647dc137f6231a9724cf65e6'
    }
    DEEPLAB_V3_DM05 = {
        'origin': '1bukSOJf8JL_RSQwrCIypvzxxamEhO9cV',
        'filename': 'deeplabv3_mnv2_dm05_voc_tainaug_os16_edgetpu.tflite',
        'checksum': '2d3ad50d08c12dba4d5ea61f59bb0b79'
    }
    DEEPLAB_V3_PLUS_DM1 = {
        'origin': '1-2U13RHX5b-h7rIfhxovpxeC4c6DNA8r',
        'filename': 'deeplabv3plus_mnv2_dm1_voc_trainaug_os4_edgetpu.tflite',
        'checksum': '3ad64d967a3e526d7df4a3b3a8a60f8a'
    }
    DEEPLAB_V3_PLUS_DM05 = {
        'origin': '1DJ11luO0SMU69egtPShP-4-rSVYki-HP',
        'filename': 'deeplabv3plus_mnv2_dm05_voc_trainaug_os4_edgetpu.tflite',
        'checksum': 'abab0449b81be44efcfab4cacccc7f1b'
    }
