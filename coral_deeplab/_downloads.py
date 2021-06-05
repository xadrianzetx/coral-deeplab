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

import os
import hashlib
import requests
from typing import Optional

from coral_deeplab.pretrained import MLModel


GOOGLE_DRIVE_EXPORT_URL = 'https://docs.google.com/uc?export=download'


def download_and_checksum_mlmodel(model: MLModel,
                                  dst: Optional[str] = None) -> str:
    """Downloads model from google drive and checks it md5sum.

    Arguments
    ---------
    model : MLModel
        One of `MLModel` options
        available in `cdl.pretrained` module.

    dst : str, default = None
        Model destination path. Saves to
        library dir if not specified

    Returns
    -------
    filepath : str
        Path to downloaded model.

    Notes
    -----
    Downloaded file will be deleted and exception raised
    if md5sum fails.
    """

    filename = model.value.get('filename')
    module_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(module_dir if not dst else dst, filename)

    if os.path.isfile(filepath):
        return filepath

    print(f'Attempting to download {filename}')
    origin = model.value.get('origin')
    response = requests.get(GOOGLE_DRIVE_EXPORT_URL, params={'id': origin})

    with open(filepath, 'wb') as file:
        checksum = hashlib.md5(response.content)
        file.write(response.content)

    if checksum.hexdigest() != model.value.get('checksum'):
        os.remove(filepath)
        raise Warning(f'md5sum failed for {filename} and file was deleted.')

    return filepath
