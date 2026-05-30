import nibabel as nib
import numpy as np
import argparse


def extract_nifti_data(img_path: str) -> np.ndarray:
    """Read the data from the nifti file and output it as a numpy array."""
    img = nib.load(img_path)
    data = img.get_fdata()
    return data


def threshold_data(data: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the data to only keep the voxels with a value above the threshold."""
    thresholded_data = data[data > threshold]
    return thresholded_data


def get_mean(data: np.ndarray) -> float:
    """Get the mean of the array."""
    return np.mean(data)


def main(img_path: str, threshold: float) -> None:
    """Print the average of the thresholded voxel values."""
    data = extract_nifti_data(img_path)
    thresholded_data = threshold_data(data, threshold)
    average = get_mean(thresholded_data)
    print(f"Mean of voxels values above {threshold}: {average}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", dest="img_path", type=str, help="Path to the nifti file.")
    parser.add_argument(
        "--thres", dest="threshold", type=float, help="Threshold to apply on the image."
    )
    args = parser.parse_args()

    main(args.img_path, args.threshold)
