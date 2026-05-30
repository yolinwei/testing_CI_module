import os
import numpy as np
import nibabel as nib
import pytest

from main import extract_nifti_data, threshold_data, get_mean

def test_get_mean():
    test_array = np.array([1, 1, 1, 1, 1])
    assert get_mean(test_array) == 1

def test_threshold_data():
    test_array = np.array([1, 2, 3, 4, 5])
    threshold = 3
    result = threshold_data(test_array, threshold)
    assert np.array_equal(result, np.array([4, 5]))

def test_extract_nifti_data(tmpdir):
    fake_data = np.ones((3, 3, 3))
    fake_img = nib.Nifti1Image(fake_data, affine=np.eye(4))
    fake_file_path = os.path.join(tmpdir, "fake_brain.nii.gz")
    nib.save(fake_img, fake_file_path)
    extracted_data = extract_nifti_data(fake_file_path)
    assert np.array_equal(extracted_data, fake_data)
