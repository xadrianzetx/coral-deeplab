import os
import unittest

import coral_deeplab as cdl
from coral_deeplab._downloads import ChecksumFailedError
from coral_deeplab._downloads import download_and_checksum_mlmodel


LIBRARY_DIR = os.path.join(os.getcwd(), "coral_deeplab")


class TestModel(cdl.pretrained.MLModel):
    TEST_VALID = {
        "origin": "1lK6os7T0BkoVWj_KhHDnryOyPvNEHR8h",
        "filename": "valid.model",
        "checksum": "0cc175b9c0f1b6a831c399e269772661",
    }
    TEST_INVALID = {
        "origin": "1b9GjjKFZ5ZIqiepsK4lVLzsTZDw2FuqJ",
        "filename": "invalid.model",
        "checksum": "foo",
    }


class TestModelDownloadModule(unittest.TestCase):
    def test_invalid_checksum(self):
        """Test if checksum"""

        with self.assertRaises(ChecksumFailedError):
            download_and_checksum_mlmodel(TestModel.TEST_INVALID)

        target_path = os.path.join(LIBRARY_DIR, "invalid.model")
        if os.path.isfile(target_path):
            self.fail("Failed to cleanup after downloading")

    def test_download_dest_default(self):
        """Test default download destination"""

        model_path = download_and_checksum_mlmodel(TestModel.TEST_VALID)
        target_path = os.path.join(LIBRARY_DIR, "valid.model")
        self.assertEqual(model_path, target_path)

    def test_download_dest_specified(self):
        """Test user specified download destination"""

        model_path = download_and_checksum_mlmodel(TestModel.TEST_VALID, dst=os.getcwd())
        target_path = os.path.join(os.getcwd(), "valid.model")
        self.assertEqual(model_path, target_path)
